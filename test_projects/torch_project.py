import contextlib
import gc
import io
import math
import os
import random
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import operator
from copy import deepcopy
from collections import OrderedDict
from itertools import product
from operator import mul
from functools import reduce
import torch

from torch import nn
from torch._six import inf, nan
from torch.autograd.function import once_differentiable
from torch.autograd.profiler import (profile, record_function, emit_nvtx)
from torch.autograd.profiler_util import (_format_time, EventList, FunctionEvent, FunctionEventAvg)
import torch.autograd.functional as autogradF
from torch.utils.checkpoint import checkpoint
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfNoLapack, slowTest, IS_WINDOWS, IS_MACOS,
    disable_gc, gradcheck, gradgradcheck, parametrize, instantiate_parametrized_tests)
from torch.autograd import Variable, Function, detect_anomaly, kineto_available
from torch.autograd.function import InplaceFunction
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, skipCUDAIfRocm,
                                                        onlyCPU, onlyCUDA, dtypes, dtypesIfCUDA,
                                                        deviceCountAtLeast, skipMeta)
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.logging_tensor import no_dispatch

import pickle

torch.distributed.algorithms.Join(joinables=1, enable=True, blabla1 = 1, blabla2 = 2 )
torch.distributed.optim.DistributedOptimizer(1, 2, 3, 4)
def graph_desc(fn):
    if fn is None:
        return 'None'
    result = type(fn).__name__ + '('
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ', '
    if next_functions:
        result = result[:-2]
    return result + ')'


class TestAutograd(TestCase):

    def test_tensor_grad_warnings(self):
        dummy = torch.empty(1)

        with warnings.catch_warnings(record=True) as w:
            # Accessing .grad on leaf
            dummy.requires_grad_()
            foo = dummy.grad
            self.assertEqual(len(w), 0)

            # Accessing .grad on non-leaf
            dummy = dummy.clone()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

            # Accessing .grad on non-leaf that retains gradients
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)

        return x, y

    def test_function(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_tensors
                # NOTE: self is the test case here
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (grad_output + grad_output * var2, None,
                        grad_output * ctx.pyscalar + grad_output * var1)

        x, y = self._function_test(MyFunction)

        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertExpected(x_grad_desc, "x_grad_desc")
        self.assertExpected(y_grad_desc, "y_grad_desc")

    def test_once_differentiable(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                self.assertFalse(torch.is_grad_enabled())
                t1, t2 = ctx.saved_tensors
                return (grad_output + grad_output * t2, None,
                        grad_output * ctx.pyscalar + grad_output * t1)

        x, y = self._function_test(MyFunction)
        self.assertEqual(graph_desc(x.grad.grad_fn),
                         'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')
        self.assertEqual(graph_desc(y.grad.grad_fn),
                         'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')

    def test_function_returns_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.))

            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.))

    def test_function_returns_undefined_tensor(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return None

        # Test that undefined tensors returned from custom backward function
        # are propagated as undefined and not tensor full of zeroes
        x = torch.ones(1, requires_grad=True)

        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x ** 2).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)

        self.assertIsNone(torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0])

    def test_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(grad, torch.zeros(1))
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_dont_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertIsNone(grad)
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_legacy_function_deprecation_exception(self):
        # Trigger exception
        class MyFunction(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        # Check exception occurs
        with self.assertRaisesRegex(
                RuntimeError,
                'Legacy autograd function with non-static forward method is deprecated'):
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):

        @staticmethod
        def forward(ctx, input):
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            raise Exception("Simulate error on backward pass")

    def test_custom_function_exception(self):

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        tmp = (t1 + t2) * (t1 + t2)
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        with self.assertRaisesRegex(Exception, "Simulate error on backward pass"):
            t3.sum().backward()

    def test_custom_function_non_tensor_inputs_outputs(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale

                # Save scale
                ctx.scale = scale
                ctx.save_for_backward(t1, t2, t3)
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                # Verify grads
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])

                scale = ctx.scale
                var1, var2, var3 = ctx.saved_tensors
                return (
                    grads[1] * scale + grads[4] * var2 * scale + grads[6],
                    grads[1] * var3 * scale + grads[4] * var1 * scale,
                    None,
                    grads[1] * var2 * scale + grads[4] * scale,
                )

        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # Test gradcheck
        def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

        gradcheck(foo, (t1, t2, t3))

    def test_custom_function_no_tensors(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                return (args[0], args[1], None, args[2])

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_invalid_gradients(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return torch.randn(10, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, 'expected shape'):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()

    def test_unrelated_inputs(self):
        # test to ensure grad(grad)check runs successfully even if there is an
        # unrelated (but differentiable) inputs

        def my_function(x, y):
            return x * x

        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)

        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

    def test_not_implemented_grad(self):
        a = torch.rand(2, requires_grad=True)
        # if grad for nextafter ends up being implemented, this should be changed
        y = torch.nextafter(a, a).sum()
        with self.assertRaisesRegex(
                NotImplementedError,
                'the derivative for .* is not implemented'):
            y.backward()

    def test_not_implemented_fwad(self):
        x = torch.randn(3)
        v = torch.rand(3)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, v)

            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                # if forward AD ends up being implemented for torch.atan2, choose a different op
                torch.atan2(dual_x, dual_x)

    def test_accumulate_grad(self):
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return x_grad, x_grad_clone

        # Accumulate in-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

        # Accumulate out-of-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)

    def test_accumulate_grad_tensor_reference(self):
        def _test_grad_tensor(params_grad_tensor, backward_grad_tensor, should_preserve_reference, create_graph):
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(id(grad_saved) == id(params.grad), should_preserve_reference)

        for create_graph in (False, True):
            # Accumulate dense gradient to sparse gradient will change the `params.grad` reference
            _test_grad_tensor(
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                torch.tensor([1.5, 1.5]),
                False,  # never accumulates in-place
                create_graph)

            # Accumulate dense gradient to dense gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph)

            # Accumulate sparse gradient to sparse gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                not create_graph,
                create_graph)

    def test_accumulate_grad_with_zero_numel_grad(self):
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        assert c.shape == (4, 0)
        c.sum().backward()

        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)

        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad, x_grad + x_hv)
        self.assertEqual(y.grad, y_grad + y_hv)

    def test_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum], grad_outputs=[torch.ones(2, 2)],
            inputs=[x], create_graph=True)
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # Test that grad_outputs and outputs have the same shape
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(
                outputs=[grad_sum], grad_outputs=[grad_out],
                inputs=[x], create_graph=True)
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(str(error), "Mismatch in shape: grad_output[0] has a shape of "
                             + str(grad_out.shape) + " and output[0] has a shape of "
                             + str(grad_sum.shape) + ".")

    def test_grad_nonleaf(self):
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            return x ** 2 + y * x + y ** 2

        for _ in range(5):
            grad_x, = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True)

            grad_x_expected = 2 * x + y
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x, grad_x_expected)

            x = x + 0.05 * grad_x

        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
        self.assertGreater(val_final, val_init)

        x.backward(grad_output)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(x_init.grad)

    def test_grad_nonleaf_many_outputs(self):
        # This checks an edge case for function callbacks
        # We want to capture two grads of a function, but can only
        # register a single callback.
        x = torch.randn(4, 2, requires_grad=True)
        a, b = x.chunk(2)

        def hook(*grads):
            hook_called[0] = True
        hook_called = [False]
        x.register_hook(hook)

        go = torch.randn(2, 2)
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True)

        self.assertEqual(grad_a, go)
        self.assertEqual(grad_b, go * 2)
        self.assertFalse(hook_called[0])
        self.assertIsNone(x.grad)

    def test_grad_nonleaf_register_hook(self):
        # This checks an edge case for register_hook.
        # We want to capture grad of a nonleaf tensor,
        # but avoid segfault during backward of other nonleaf tensors
        x = torch.randn(5, requires_grad=True)
        x_list = x.unbind()

        x0 = x_list[0]
        hook_results = [None]

        def hook(grad):
            hook_results[0] = grad
        x0.register_hook(hook)

        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.))
        expected_grad = torch.tensor([1., 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)

        for i in range(1, 5, 1):
            x_list[i].backward()
            self.assertEqual(hook_results[0], None)
            expected_grad[i] = 1.0
            self.assertEqual(x.grad, expected_grad)
            self.assertIsNone(x_list[i].grad)

    def test_hook_with_no_name(self):
        # Create a hook that do not have a __name__ attribute
        class MyHookClass:
            def __call__(self, grad):
                return grad.clone()

        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # Should run fine

    def test_sharded_grad(self):
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]
        intermediates = [l * i + l * l for i, l in enumerate(leaves)]
        loss = sum(v * i for i, v in enumerate(intermediates)).sum()

        # define a helper for dividing intermediates into groups
        def group(l, group_size):
            return (l[i:i + group_size] for i in range(0, len(l), group_size))

        # Compute the d loss / d intermediates in chunks of shard_size
        shard_size = 2
        d_intermediates = [d_i for intermediates_batch in group(intermediates, shard_size)
                           for d_i in torch.autograd.grad(loss, intermediates_batch)]
        # Compute rest of backward pass
        torch.autograd.backward(intermediates, d_intermediates)

        for i, l in enumerate(leaves):
            self.assertEqual(l.grad, i * i * (1 + l))

    def test_backward_badcalls(self):
        x = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            x.backward()

    def test_grad_badcalls(self):
        x = torch.ones(1)
        y = x ** 2
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            torch.autograd.grad(x, y)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            torch.autograd.grad(y, x)

        x = torch.ones(1, requires_grad=True)
        y = x ** 2
        torch.autograd.grad(y, x)  # this should succeed now

    def test_grad_empty_inputs(self):
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "grad requires non-empty inputs."):
            torch.autograd.grad(2 * x, [], grad_outputs=torch.tensor([1.0]))

    def test_grad_fn_badcalls(self):
        error_regex = 'expected .* arguments, got .* instead'
        x = torch.ones(1, requires_grad=True)
        y = x ** 2
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())  # too many
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()  # too few

        y.grad_fn(x.detach())  # this should succeed

    def test_grad_unreachable(self):
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # Make sure x and y have grad accumulators allocated
        z = x * 2
        w = y * 2

        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # This is slightly different than the case above, because z doesn't even
        # have a grad accumulator allocated.
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

        # allow_unused=False, but grads contains None inside, should throw
        with self.assertRaisesRegex(RuntimeError,
                                    "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

    def test_grad_unreachable_discovery(self):
        # Test that certain nodes are not erroneously executed when an input
        # is unreachable. See #39784
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                self.fail("This node should not be executed!")

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        (gY,) = torch.autograd.grad(x, (y, ), allow_unused=True)
        self.assertIsNone(gY)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y, ))  # allow_unused is implicitly True!
        self.assertIsNone(y.grad)

    def test_grad_batched_grad(self):
        x = torch.randn(2, 2, requires_grad=True)

        out = x.clone()  # Size([2, 2])
        batched_grad = torch.arange(3).expand(2, 2, 3).transpose(0, 2)  # Size([3, 2, 2])
        grad, = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype))

        # Detect shape mismatch
        grad_out = torch.ones(2, 2)
        with self.assertRaisesRegex(RuntimeError, "If `is_grads_batched=True`, we interpret the first"):
            torch.autograd.grad(outputs=out, grad_outputs=(grad_out,), inputs=(x,), is_grads_batched=True)

        # Scalar outputs
        out = x.sum()  # Size([])
        batched_grad = torch.arange(3)  # Size([3])
        grad, = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype))

        # We consider scalar and sized-1 to be a mismatch. This is consistent with current non-batched behavior.
        grad_out = torch.ones(2).unsqueeze(1)
        with self.assertRaisesRegex(RuntimeError, "If `is_grads_batched=True`, we interpret the first"):
            torch.autograd.grad(outputs=out, grad_outputs=(grad_out,), inputs=(x,), is_grads_batched=True)

    def test_hooks(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

        counter = [0]

        def bw_hook(inc, grad):
            self.assertIsInstance(grad, torch.Tensor)
            counter[0] += inc

        z = x ** 2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)

        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def test_hooks_cpp(self):
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()

        self.assertEqual(counter[0], 1, msg='bw_hook not called')
        self.assertEqual(x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0)

    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                assert ctx.needs_input_grad[0]
                assert not ctx.needs_input_grad[1]
                return x, y

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                return grad_x, None

        was_called = [False]

        def hook(grad):
            self.assertIsNotNone(grad)
            was_called[0] = True

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        rx, ry = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()

        # It should be possible to call retain_grad() multiple times
        h1.retain_grad()
        h1.retain_grad()

        # Gradient should be accumulated
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    def test_retain_grad_cycle(self):
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            y = x * 2
            y.retain_grad()

            return y / 2, torch._C._WeakTensorRef(y)

        z, ref = run_test()
        self.assertTrue(ref.expired())
        z.sum().backward()

    def test_backward(self):
        v = torch.randn(5, 5, requires_grad=True)
        x = torch.randn(5, 5, requires_grad=True)
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        z = torch.randn(5, 5, requires_grad=True)
        grad_output = torch.randn(5, 5)

        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        a = x + (y * z) + 4 * z ** 2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_sparse_mm_backward(self):
        size = (3, 3)
        sparse = torch.sparse_coo_tensor(size, requires_grad=True)
        dense = torch.randn(size, requires_grad=True)

        with self.assertRaisesRegex(
                RuntimeError,
                "The backward pass for this operation requires the 'mat1' tensor to be strided,"):
            z = dense.addmm(sparse, dense)

        mm_test_cases = [
            # a requires grad, a is sparse, b requires grad, b is sparse, error message
            (False, True, True, False, None),
            (False, False, True, True, "The backward pass for this operation requires the 'mat2'"),
            (False, True, True, True, "The backward pass for this operation requires the 'mat2'"),
            (True, False, True, True, "The backward pass for this operation requires the 'mat2'"),
            (True, True, False, False, "The backward pass for this operation requires the 'self'"),
            (True, True, True, False, "The backward pass for this operation requires the 'self'"),
            (True, True, True, True, "The backward pass for this operation requires the 'mat2'"),
        ]
        for a_req_grad, a_is_sparse, b_req_grad, b_is_sparse, err_msg in mm_test_cases:
            # We should only be testing cases with sparse inputs, and at least one
            # input needs to require grad so we can call a backward pass
            assert a_is_sparse or b_is_sparse
            assert a_req_grad or b_req_grad

            a = torch.randn(size, requires_grad=a_req_grad)
            if a_is_sparse:
                a = a.to_sparse()
            b = torch.randn(size, requires_grad=b_req_grad)
            if b_is_sparse:
                b = b.to_sparse()

            # If no error expected, check that sparse and dense cases match
            if err_msg is None:
                r = a.mm(b)
                r.sum().backward()
                a_grad = None if a.grad is None else a.grad.clone().detach()
                b_grad = None if b.grad is None else b.grad.clone().detach()

                # Redo with only dense tensors
                a = (a.to_dense() if a.is_sparse else a).clone().detach()
                a.requires_grad = a_req_grad
                b = (b.to_dense() if b.is_sparse else b).clone().detach()
                b.requires_grad = b_req_grad
                r = a.mm(b)
                r.sum().backward()

                self.assertEqual(a_grad, a.grad)
                self.assertEqual(b_grad, b.grad)

            else:
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mm(b)

    def test_multi_backward(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q = torch.randn(5, 5, requires_grad=True)

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        torch.autograd.backward([z, c], [grad_z, grad_c])

        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)

        z = x + y
        q = y * 2

        # NB: we currently raise an exception if any arguments to backwards
        # have requires_grad=False and don't have a grad_fn. We may want to
        # relax that check to a warning.
        def call_backwards():
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])
        self.assertRaises(RuntimeError, call_backwards)

    def test_backward_with_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        def fn():
            return x ** 2 + y * x + y ** 2

        gradient = torch.ones(2, 2)
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        @torch.no_grad()
        def reset_grad():
            x.grad.zero_()
            y.grad.zero_()

        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, y_grad_expected)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[x])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[y])
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=y)
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        self.assertRaisesRegex(RuntimeError, 'cannot be empty',
                               lambda: torch.autograd.backward(fn(), gradient, inputs=[]))

    def test_backward_with_nonleaf_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        out = x_nonleaf ** 2 + y * x_nonleaf + y ** 2

        out.backward(torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[x, y, x_nonleaf])
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y
        x_non_leaf_expected = 2 * x_nonleaf + y

        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(x_nonleaf.grad, x_non_leaf_expected)

        # backward doesn't have an allow_unused flag, so the behavior of backward
        # when variable is not part of the graph is as if allow_used were true
        # x.grad will simply be None.
        out.backward(torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z])
        self.assertIsNone(z.grad)

    def test_dependent_backward(self):
        x = torch.randn(10, requires_grad=True)
        y = x ** 2
        z = y ** 3

        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])

        xd = x
        self.assertEqual(x.grad, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

    def test_save_output_nr(self):
        x = torch.randn(10, requires_grad=True)

        class MultiOutputFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:5], x[5:]

            @staticmethod
            def backward(ctx, *grad):
                return torch.cat(grad)

        a, b = MultiOutputFn.apply(x)
        self.assertEqual(b.output_nr, 1)

        class TestFn(Function):
            @staticmethod
            def forward(ctx, b):
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                b, = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)

        TestFn.apply(b).sum().backward()

    def test_free_deep_graph(self):
        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build a "chain" computation graph
            for _ in range(depth):
                y = y + y * 0.000001

            # graph deletion occurs when the above locals go out of scope.
            # In this case `del y` will trigger it but it's easier to leave
            # it to Python to delete the locals.

        # Should not stack overflow
        scope()

    def test_free_deep_graph_complicated(self):
        def scope():
            depth = 100000
            randchoice = torch.randint(2, [depth, 2])
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # Hold the two previous values
            prev_values = [None, None]

            # Build a "chain with skip connections" graph
            for _ in range(depth):
                prev_tensors = [tensor for tensor in prev_values[:-1]
                                if tensor is not None]
                prev_values.append(y)
                prev_values.pop(0)

                # Definitely pick one tensor to add
                y += y * 0.000001

                # Possibly add other tensors
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_free_deep_graph_pyfunction(self):
        class MyOp(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build deeply nested computation graph
            for _ in range(depth):
                y = MyOp.apply(y, y)

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_no_unnecessary_save(self):
        # If we kept x in the derivative Function of x * 2 we would
        # get an error in the backward that would complain that we've
        # modified x, which was needed for gradient computation.
        # Since we should elide unnecessary saves, this test should pass.
        mu = torch.ones(1, requires_grad=True)
        x = torch.empty(1)
        loss = 0
        for i in range(3):
            x.detach_()
            x.copy_(mu + i)
            ft = torch.tensor([float(i)])
            multiplied = x * ft
            s = multiplied.sum()
            loss += s
        loss.backward()

    def test_no_grad(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        with torch.no_grad():
            w = x + y

        @torch.no_grad()
        def adder(x, y):
            return x + y

        z = adder(x, y)

        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.grad_fn)
        self.assertFalse(z.requires_grad)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))
        self.assertIsNone(z.grad_fn)

        # test nested decorator and with-statement on no_grad
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            w = adder(x, y)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_generator_functions(self):
        @torch.no_grad()
        def gen_no_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i

        with torch.enable_grad():
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i

        with torch.no_grad():
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        # enable_grad_decorator_recursive and no_grad_decorator_recursive call each other
        # recursively, to ensure that the decorators preserve the caller's setting
        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        # enable_grad_context_manager_recursive and no_grad_context_manager_recursive call
        # each other recursively, to ensure that the decorators preserve the caller's setting
        def enable_grad_context_manager_recursive(depth):
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_coroutines(self):
        @torch.no_grad()
        def coro_no_grad(n=10):
            self.assertFalse(torch.is_grad_enabled())
            for i in range(n):
                self.assertFalse(torch.is_grad_enabled())
                r = yield i
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertFalse(torch.is_grad_enabled())

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            self.assertTrue(torch.is_grad_enabled())
            for i in range(n):
                self.assertTrue(torch.is_grad_enabled())
                r = yield i
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertTrue(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            coro, r = coro_no_grad(), None
            try:
                while True:
                    self.assertTrue(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertTrue(torch.is_grad_enabled())

            except StopIteration:
                pass

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            coro, r = coro_enable_grad(), None
            try:
                while True:
                    self.assertFalse(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertFalse(torch.is_grad_enabled())

            except StopIteration:
                pass

    def test_set_grad_coroutines_benign_exceptions(self):
        class RecoverableException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True

        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        class UnrecoverableException(Exception):
            pass

        class SecondaryException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException

        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

    def test_set_grad_coroutines_exit(self):
        @torch.no_grad()
        def coro_no_grad(state):
            for i in range(10):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertFalse(torch.is_grad_enabled())
                    state.add('GeneratorExit')
                    raise

        @torch.enable_grad()
        def coro_enable_grad(state):
            for i in range(10):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertTrue(torch.is_grad_enabled())
                    state.add('GeneratorExit')
                    raise

        state = set()
        with torch.enable_grad():
            coro = coro_no_grad(state)
            for i in range(5):
                next(coro)

            coro.close()
        self.assertTrue('GeneratorExit' in state)

        state = set()
        with torch.no_grad():
            coro = coro_enable_grad(state)
            for i in range(5):
                next(coro)

            coro.close()
        self.assertTrue('GeneratorExit' in state)

    def test_no_grad_python_function(self):
        """Python Functions should respect grad mode."""
        x = torch.ones(5, 5, requires_grad=True)

        class MyOp(Function):
            @staticmethod
            def forward(self, x):
                return x + 1

            @staticmethod
            def backward(self, dy):
                return dy

        with torch.no_grad():
            y = MyOp.apply(x)
        self.assertFalse(y.requires_grad)

    def test_indexing(self):
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)

            indexed_var.sum().backward()
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)

        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([slice(None), [2, 3]]))
        check_index(x, y, ([[2, 3], slice(None)]))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0]))
        check_index(x, y, ([0], ))

        x = torch.arange(1., 49).view(4, 3, 4)
        y = Variable(x, requires_grad=True)

        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0], ))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))

        # advanced indexing, with a tensor wrapped in a variable
        z = torch.LongTensor([0, 1])
        zv = Variable(z, requires_grad=False)
        seq = [z, Ellipsis]
        seqv = [zv, Ellipsis]

        if y.grad is not None:
            with torch.no_grad():
                y.grad.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self):
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad, expected_grad)

        # with advanced indexing
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = [[1, 1, 3, 2, 1, 2], [0]]
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = [[[1, 2], [0, 0]], [[0, 1], [1, 1]]]
        y[idx].sum().backward()
        expected_grad = torch.tensor([[0., 2., 0., 0.],
                                      [1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 0., 0.]])
        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1., 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        idx = [[1, 1, 1], slice(None), slice(None)]
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        # Example from https://github.com/pytorch/pytorch/issues/24853.
        # if `index(tensor, indices)` saves `tensor` for backwards, then it will
        # trigger a version check on `tensor` during the backward pass, which
        # will cause the following code to error because `tensor` gets modified
        # by the indexing line.
        a = torch.tensor([1., 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        tensor[a != 0] = tensor[a != 0]
        tensor.backward(torch.zeros_like(tensor))

    def test_volatile_deprecated(self):
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(v.volatile)
        self.assertIn('volatile', str(w[0].message))

    def test_saved_variables_deprecated(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_variables
                return (grad_output, grad_output)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            MyFunction.apply(x, y).sum().backward()

            has_deprecated = map(lambda warn:
                                 'deprecated' in str(warn) and
                                 'saved_variables' in str(warn),
                                 warns)
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            self.assertTrue(has_deprecated)

    def test_requires_grad(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)

        def error():
            raise RuntimeError
        # Make sure backward isn't called on these
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks['test'] = error
        x._backward_hooks['test'] = error
        y._backward_hooks['test'] = error
        b.backward(torch.ones(5, 5))

    def test_requires_grad_(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        z = x * y
        self.assertRaises(RuntimeError, lambda: z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)

        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    def test_requires_grad_inplace(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

        # non-leaf
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

    def test_no_requires_grad_inplace(self):
        # basic case, should be able to modify inplace while requires_grad is False
        a = torch.randn(2, 3)
        a.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # same but with a view
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # should fail if requires_grad = True when we modify inplace
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_attribute_deletion(self):
        x = torch.randn((5, 5), requires_grad=True)
        del x.grad
        self.assertIsNone(x.grad)
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    def test_duplicate_backward_root(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])

        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    def test_backward_no_grad(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_backward_twice_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: c.backward(torch.tensor([1, 1, 1], dtype=torch.double)))

    def test_backward_twice_retained_graph_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = b + 1
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_retained_graph_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_create_graph_warns(self):
        try:
            prev = torch.is_warn_always_enabled()
            torch.set_warn_always(True)

            b = torch.randn(3, requires_grad=True, dtype=torch.double)
            c = b * b
            with warnings.catch_warnings(record=True) as ws:
                c.backward(torch.ones_like(c), create_graph=True)
            b.grad = None
            self.assertTrue(any('Using backward() with create_graph=True' in str(w.message) for w in ws))

            # Should not warn for grad
            with warnings.catch_warnings(record=True) as ws:
                torch.autograd.grad(c, b, torch.ones_like(c), create_graph=True)
            self.assertFalse(any('Using backward() with create_graph=True' in str(w.message) for w in ws))

        finally:
            torch.set_warn_always(prev)

    def test_next_functions(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        self.assertIsNotNone(a.grad_fn)
        next_functions = a.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[0][1], 0)
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[1][1], 0)

        b = a + 5
        next_functions = b.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIs(next_functions[0][0], a.grad_fn)
        self.assertIs(next_functions[1][0], None)

    def test_inplace(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_graph=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_graph=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_graph=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    def test_mark_non_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 0).to(torch.double)

        x = torch.randn(5, 5, requires_grad=True)
        mask = MyFunction.apply(x)
        self.assertFalse(mask.requires_grad)
        y = x.masked_fill(mask, 0)
        y.sum().backward()

    def test_mark_non_differentiable_mixed(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                a = input + 1
                b = input + 2
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                self.assertTrue((grad_a == 0).all())
                self.assertTrue((grad_b == 1).all())
                return grad_b

        x = torch.randn(5, 5, requires_grad=True)
        a, b = MyFunction.apply(x)
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        # This used to segfault because MyFunction would send back null
        # gradients to MulBackward, which is implemented in C++. C++
        # implemented functions expect incoming  grad_ouptuts to be non-null.
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input.clone()
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return None

        x = torch.randn(5, 5, requires_grad=True)
        r = MyFunction.apply(x * x)
        (r * x).sum().backward()

    def test_return_duplicate(self):
        class DoubleDuplicate(Function):
            @staticmethod
            def forward(ctx, x):
                output = x * 2
                return output, output

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def fn(x):
            a, b = DoubleDuplicate.apply(x)
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(fn, [x])
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        class DoubleInplace(Function):
            @staticmethod
            def forward(ctx, x):
                x.mul_(2)
                ctx.mark_dirty(x)
                return x, x

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def inplace_fn(x):
            a, b = DoubleInplace.apply(x.clone())
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(inplace_fn, [x])
        gradgradcheck(inplace_fn, [x])

        # Can't modify leaf variables in-place
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x))
        # Functions which modify views in-place must return only one output
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x.clone()[0]))

    def _test_setitem(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones_like(value))

        # case when x broadcasts to as y[1]
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        y[1] = x
        y.backward(torch.randn(2, 3, 4))
        self.assertEqual(x.size(), x.grad.size())

    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), [[0, 4, 2]])
        self._test_setitem((5, 5), [[0, 4], [2, 2]])
        self._test_setitem((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), [[0, 1], [1, 0]])
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor((5,), Variable(torch.LongTensor([3]), requires_grad=False).sum())
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [Variable(torch.LongTensor([1,
                                              3]), requires_grad=False), [2, 4], slice(None)])

    def test_setitem_mask(self):
        mask = torch.BoolTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # both select and sum return Scalars in ATen; ensure they work together.
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            return x.select(0, 1).sum()

        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        value = torch.rand([])
        v_expanded = torch.tensor(value).expand(10)
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        result, = torch.autograd.grad(a.diagonal(), a, v_expanded)
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        v_expanded = torch.rand(10).expand(10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        result, = torch.autograd.grad(a[0], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        result, = torch.autograd.grad(a[3:5], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        self.assertEqual(result, expected)

    def test_unused_output(self):
        x = torch.randn(10, 10, requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad, expected_grad)

        with torch.no_grad():
            x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    def _test_sparse_gather(self, size_x, size_ind, dim):
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        out = torch.gather(x, dim, ind, sparse_grad=False)
        grad = torch.rand_like(out)
        out.backward(grad)
        grad_dense = x.grad.clone()
        x.grad = None
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        self.assertEqual(grad_dense, x.grad.to_dense())

    def test_sparse_gather_dim0(self):
        self._test_sparse_gather((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        self._test_sparse_gather((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        self._test_sparse_gather((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        self._test_sparse_gather((), (), 0)

    def test_gc_in_destructor(self):
        """
        Previously, if a Function destructor triggered a garbage collection,
        the Variable's tp_dealloc handler would get called twice leading to a
        segfault.
        """
        class CollectOnDelete(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

            def __del__(self):
                gc.collect()

        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    def test_naughty_autograd_function_attribute_access(self):
        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_x):
                return grad_x

        with self.assertWarnsRegex(DeprecationWarning, "should not be instantiated"):
            f = Id()

        # # After raising warning, should still return an instance
        self.assertIsInstance(f, Id)
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "non-static forward method is deprecated"):
            f(x)
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), "IdBackward")

        # THPFunction is the base class of both grad_fn and autograd functions,
        # which means that a lot of accessors on them may segfault. Test that we
        # properly error in this case.
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = dict()
        with self.assertRaisesRegex(RuntimeError, "Attribute '_register_hook_dict' is invalid"):
            f._register_hook_dict(t)
        with self.assertRaisesRegex(RuntimeError, "Attribute 'register_hook' is invalid"):
            f.register_hook(lambda x, y: None)
        with self.assertRaisesRegex(RuntimeError, "Attribute 'next_functions' is invalid"):
            f.next_functions
        with self.assertRaisesRegex(RuntimeError, "Attribute 'name' is invalid"):
            f.name()
        with self.assertRaisesRegex(RuntimeError, "underlying PyNode has already been deallocated"):
            f.metadata

    @unittest.expectedFailure
    def test_naughty_anomaly_access(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                return g

        x = torch.zeros(1, requires_grad=True)
        y = MyFunction.apply(x)
        y.backward()
        y.grad_fn.metadata
        g = y.grad_fn
        del y
        g.metadata  # this currently fails, but shouldn't

    def test_naughty_autograd_function_stashing_ctx(self):
        saved_ctx = []

        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                saved_ctx.append(ctx)
                return ctx.saved_tensors

        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        # At this point in time, it complains that the graph has been freed
        # (which indeed true, although a somewhat indirect way of stating the
        # problem).
        self.assertRaises(RuntimeError, lambda: saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        # This test failed the equality check in PR #22983; it's an interesting
        # and different test case worth enshrining.  mult1 is not testing
        # anything that interesting, but mult2 is the interesting case.

        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            gy, = torch.autograd.grad(y[0], x, create_graph=True)
            ggy_1, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            gy, = torch.autograd.grad(y[0], x, create_graph=True)
            ggy_2, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        # This test failed complaining that buffers had already been freed
        # prior to #22983.  Also pretty interesting test case.
        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x ** 2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))

        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)  # should not error!

    def test_detach(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)

        # in-place deatch on a view raises an exception
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, 'view', lambda: view.detach_())

    def test_detach_base(self):
        "detaching base does not detach view"
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)

    def test_detach_then_inplace_raises_in_autograd(self):
        x = torch.randn([], requires_grad=True)
        orig_x = x.detach().clone()

        y = x ** 2  # saves x
        z = x.detach()
        z.zero_()
        with self.assertRaisesRegex(RuntimeError, "has been modified by an inplace"):
            y.backward()

    def test_detach_disallows_metadata_change(self):
        x = torch.randn([], requires_grad=True)
        detached = x.detach()

        with self.assertRaisesRegex(
                RuntimeError, "not allowed on a Tensor created from .data or .detach()"):
            detached.resize_(3, 3)

    def _test_type_conversion_backward(self, t, ):
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)

        for t in [torch.DoubleTensor, torch.FloatTensor, torch.IntTensor, torch.ByteTensor]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            _, y_type = y_c.type().rsplit('.', 1)
                            y_typestr = ('torch.cuda.' if y_cuda else 'torch.') + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(y_c.data_ptr(), y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr())

        self._test_type_conversion_backward(lambda x: x)
        if torch.cuda.is_available():
            self._test_type_conversion_backward(lambda x: x.cuda())
            if torch.cuda.device_count() >= 2:
                # one of these has to be the non-default device
                self._test_type_conversion_backward(lambda x: x.cuda(0))
                self._test_type_conversion_backward(lambda x: x.cuda(1))

    def test_isolated_node(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        o = (b + a).sum()
        o.backward()

    def test_shape(self):
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        with torch.no_grad():
            x.numpy()

        x = torch.randn(2, 2)
        x.numpy()

        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        class Identity(Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Identity.apply(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):
            @staticmethod
            def forward(ctx, a, b):
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Inplace.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y

        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation',
                               lambda: z.backward())

    def test_no_grad_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(self, x):
                return x

            @staticmethod
            def backward(self, grad_output):
                return grad_output

        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)

        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        # This tests checks backward engine for a very subtle bug that appreared
        # in one of the initial versions of autograd. Gradients tensors were
        # simply stored in lists while the function waited for all its gradients
        # to be computed. However, sometimes an output was used multiple times,
        # so the gradients needed to be summed. Engine used to keep a need_copy
        # set of tensors that will need a clone upon next addition and removed
        # them from the set as soon as the clone was performed. However, this
        # could lead to incorrect results if the same gradient tensor was
        # buffered in three places in the graph:
        # 1. When accumulating gradients in one of these places it was cloned
        #    and removed from need_copy set.
        # 2. When accumulating in second place, it wasn't in the need_copy set,
        #    so the gradients were simply accumulated in-place (which already
        #    modified the grad in 3rd place)
        # 3. When accumulating in the third place, it wasn't in the need_copy set
        #    as well, so the incoming gradient was summed in-place, yielding
        #    incorrect results in all functions, except the first one.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # Simulate that we're in the middle of the graph
        a = x + 2
        b = y + 2
        c = x + 2
        # This op will just return grad_output two times in backward
        add1 = a + b
        add2 = add1 + c
        # Simulate a long branch, so grad_output will get buffered.
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # expected gradients are:
        # for x: 34 (16 from final a, 16 from final c, 2 from add2)
        # for y: 17 (16 from final b, 1 from add2)
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        test_case = self

        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    def test_dep_nograd(self):
        class F1(Function):
            @staticmethod
            def forward(ctx, input):
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return input, out

            @staticmethod
            def backward(ctx, grad_output, ignored):
                return grad_output

        class F2(Function):
            @staticmethod
            def forward(ctx, input, ignored):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        x = torch.randn(5, requires_grad=True)
        a, b = F1.apply(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        x = torch.tensor([1.], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_simple_reentrant(self):
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # Parent graph.
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

        # Reentrant child graph.
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        f = TestAutograd.SimulateBackwardError.apply(e)
        reentrant_root = f.sum()

        class ReentrantFunc(Function):

            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will throw an error.
                reentrant_root.backward()
                return grad

        d = ReentrantFunc.apply(c)
        with self.assertRaisesRegex(Exception, 'Simulate error'):
            d.sum().backward()

    def test_var_mean_differentiable(self):
        dim = [2, 4]
        keepdim = False
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        input2 = deepcopy(input1)
        var1, mean1 = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        var2 = input2.var(dim=dim, keepdim=keepdim)
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)

        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        self.assertEqual(r1, r2, rtol=0.01, atol=0.0)

        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        self.assertEqual(input1.grad, input2.grad, rtol=0.01, atol=0.0)

    @skipIfNoLapack
    def test_lobpcg(self):

        def func(k, A, largest=True, B=None):
            X_shape = list(A.shape)
            X_shape[-1] = k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            if A.dim() > 2:
                X = X.expand(X_shape)

            D, U = torch.lobpcg(A=A, k=k, B=B, X=X, largest=largest)

            # LOBPCG uses a random initial eigenspace approximation
            # if parameter `X` is not provided.
            # This may cause a non-deterministic behavior
            # when it comes to the sign of an eigenvector
            # (note if v is an eigenvector, so is -v),
            # hence we eliminate this non-determinism
            # by making sure that each column of U
            # gets multiplied by the sign of its max (in absolute value) element.
            # Also, gradcheck changes the content of the input by +/- eps (default to 1e-06)
            # to compute the numerical gradient which can also cause the signs to flip.
            _, idx = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return D, U

        def run_symeig_test(k, sizes, largest=True):
            A = torch.rand(*sizes).double()
            A = (A @ A.mT) / 10
            A.requires_grad_(True)

            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)

            # Custom gradient vectors for better stability due to some
            # non-determinism in the lobpcg's forward.
            # Note it is not required if symeig is in forward instead (tested).
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(lambda A: func(k, A, largest), A, [D_grad, U_grad], atol=1e-4, check_batched_grad=False)

            # check whether A.grad is symmetric
            A = A.detach().requires_grad_(True)
            D, U = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.mT)

        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)

    def test_variable_traverse(self):
        def get_out_and_unrefed_cycle():
            inp = torch.randn(10, requires_grad=True)
            tmp = inp.view(10, 1)
            out = tmp.view(10)

            # Create a reference cycle that contains an
            # intermediary Variable in the graph
            my_list = []
            my_list.append(tmp)
            my_list.append(my_list)

            return out

        out = get_out_and_unrefed_cycle()
        gc.collect()
        # This will segfault if things have been erroneously released
        out.backward(torch.randn(out.size()))

    def test_pow_zero_tensor_gradient(self):
        def run_test(input_size, exponent):
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.abs().sum(), 0)

        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    def test_profiler(self):
        x = torch.randn(10, 10)

        with profile(use_kineto=kineto_available()) as p:
            self.assertTrue(torch.autograd._profiler_enabled())
            y = x * 2 + 4

        self.assertFalse(torch.autograd._profiler_enabled())

        names = ['aten::mul', 'aten::add']
        found_indices = set()
        for evt in p.function_events:
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        self.assertEquals(len(found_indices), len(names))

    def test_profiler_seq_nr(self):
        with profile(use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = z.sum()
            s.backward()
        print(p.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1))
        # expecting aten::add, aten::sum to have the sequence numbers,
        # expecting the corresponding backward nodes to have the same numbers
        # as the forward ops
        add_seq_nr = -1
        sum_seq_nr = -1
        found_add = found_sum = False
        found_bwd_add = found_bwd_sum = False
        found_empty = False
        for e in p.function_events:
            # Ignore record_function user scope.
            if "autograd::engine::evaluate_function" in e.name:
                continue
            if e.name == "aten::add":
                add_seq_nr = e.sequence_nr
                self.assertFalse(found_add)
                found_add = True
            elif e.name == "aten::sum":
                sum_seq_nr = e.sequence_nr
                self.assertFalse(found_sum)
                found_sum = True
            elif "Add" in e.name and "Backward" in e.name:
                self.assertEqual(e.sequence_nr, add_seq_nr)
                self.assertFalse(found_bwd_add)
                found_bwd_add = True
            elif "Sum" in e.name and "Backward" in e.name:
                self.assertEqual(e.sequence_nr, sum_seq_nr)
                self.assertFalse(found_bwd_sum)
                found_bwd_sum = True
            # check that nested ops (e.g. empty) don't have
            # sequence number
            if e.name == "aten::empty":
                self.assertEqual(e.sequence_nr, -1)
                found_empty = True
        self.assertGreaterEqual(add_seq_nr, 0)
        self.assertGreaterEqual(sum_seq_nr, 0)
        self.assertNotEqual(add_seq_nr, sum_seq_nr)
        self.assertTrue(found_add)
        self.assertTrue(found_sum)
        self.assertTrue(found_bwd_add)
        self.assertTrue(found_bwd_sum)
        self.assertTrue(found_empty)

    def test_profiler_unboxed_only(self):
        x = torch.rand(3, 4)

        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            x.resize_([3, 2])

    def test_profiler_propagation(self):
        def foo(x):
            with record_function("in_foo") as rf:
                return x * 2

        x = torch.rand(3, 4)
        traced_foo = torch.jit.trace(foo, x)

        def bar(x):
            with record_function("in_bar") as rf:
                # we expect that profiler will be able
                # propagate across fork
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                # note: continuation (and rf's end) can
                # be executed in a different thread
                with record_function("in_bar_after_wait") as rf2:
                    y = y * 2
                return y

        traced_bar = torch.jit.trace(bar, x)

        with profile(use_kineto=kineto_available()) as p:
            traced_bar(x)

        found_foo = False
        found_bar = False
        found_bar_after_wait = False
        for info in p.function_events:
            if info.name == "in_foo":
                self.assertFalse(found_foo)
                found_foo = True
            elif info.name == "in_bar":
                self.assertFalse(found_bar)
                found_bar = True
            elif info.name == "in_bar_after_wait":
                self.assertFalse(found_bar_after_wait)
                found_bar_after_wait = True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    def test_record_function_callbacks(self):
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            with record_function("foo"):
                y = x * 2 + 4

        function_events = p.function_events
        foo_event = [event for event in function_events if "foo" in event.name][0]
        self.assertEqual(foo_event.count, 1)

    def test_profiler_aggregation_fake(self):
        events = EventList()
        id = [0]

        def get_id():
            id[0] = id[0] + 1
            return id[0]

        # [[thread_id, [(start, end, id), ....]], ...]
        # Using list instead of a dict so order is guaranteed for any Python
        # version
        threads = [
            [1, [(0, 1, get_id()), (1, 2, get_id())]],
            [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]],
        ]
        for thread, ranges in threads:
            for range in ranges:
                assert(len(range) == 3)
                events.append(
                    FunctionEvent(
                        id=range[2],
                        node_id=0,
                        name="",
                        thread=thread,
                        start_us=range[0],
                        end_us=range[1],
                    )
                )

        events._populate_cpu_children()

        # Note that [1, 3] pushes out [0, 2] first. Then we record [1, 2]
        # as a child of [1, 3]
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            return [child.id for child in event.cpu_children]

        assert([get_children_ids(event) for event in events] == res)

    def test_profiler_aggregation_table(self):
        """
        Test if the profiling result is aggregated for `str(prof)`
        See: https://github.com/pytorch/pytorch/issues/37500
        """

        x = torch.randn(1024)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum("i->", x)

        prof_str = str(prof)
        prof_table = prof.table()

        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        avg = FunctionEventAvg()
        avg.add(FunctionEvent(id=0, node_id=0, name="foo", thread=0, start_us=10, end_us=15))
        avg.add(FunctionEvent(id=1, node_id=0, name="foo", thread=0, start_us=20, end_us=30))
        avg.add(avg)
        self.assertEqual(avg.key, "foo")

        # aggregate stats
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.cuda_time_total, 0)

        # average stats
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.cuda_time_total, 0)

    def test_profiler_shapes(self):
        print("")
        layer1 = torch.nn.Linear(20, 30)
        layer2 = torch.nn.Linear(30, 40)
        input = torch.randn(128, 20)
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            layer2(layer1(input))

        print(prof.function_events)

        linear_expected_shapes = [
            [[128, 20], [30, 20], [30]],
            [[128, 30], [40, 30], [40]],
        ]

        found_indices = set()
        for event in prof.function_events:
            if event.name == "aten::linear":
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    def test_profiler_aggregation_lstm(self):
        print("")
        rnn = torch.nn.LSTM(10, 20, 2)
        total_time_s = 0
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            for i in range(20):
                input = torch.randn(5, 3, 10)
                h = torch.randn(2, 3, 20)
                c = torch.randn(2, 3, 20)
                start = time.time()
                rnn(input, (h, c))
                end = time.time()
                total_time_s += end - start

        print(prof.table(
            sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=10))
        print(prof.table(
            sort_by="self_cpu_time_total", row_limit=10, max_src_column_width=300, header="TEST", top_level_events_only=True))
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=True))

        total_time_us = total_time_s * 1000.0 * 1000.0  # make it us which is profiler default
        print(
            "Total time based on python measurements: ",
            _format_time(total_time_us)
        )
        print(
            "CPU time measurement python side overhead: {:.2f}%".format(
                (total_time_us / prof.self_cpu_time_total - 1.0) * 100.0
            )
        )

        if sys.platform != "win32":
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

    def test_record_function(self):
        x = torch.randn(10, 10)

        def forward(x):
            with record_function("outer"):
                y = x * 2 + 4
                with record_function("inner"):
                    y = y - 1
            y = y / 1

        forward(x)

        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        events = p.function_events
        important_events = [
            'outer',
            'aten::mul',
            'aten::add',
            'inner',
            'aten::sub',
            'aten::div'
        ]
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        # We can also use record_function to decorate arbitrary function
        @record_function('my_func')
        def f(x, y):
            return x + y

        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        self.assertTrue('my_func' in str(p))

    def test_record_function_multithreaded(self):
        rf = record_function("outer")
        rf.__enter__()
        with record_function("inner"):
            # test that exiting the record function after starting another one
            # doesn't throw.
            rf.__exit__(None, None, None)

        with record_function("inner"):
            rf.__enter__()
        # test that exiting the record function after ending another one
        # doesn't throw.
        rf.__exit__(None, None, None)


    def test_dir(self):
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn('shape', keys)

        # real and imag are only implemented for complex tensors.
        y = torch.randn(10, 10, dtype=torch.cfloat)
        imag_key = 'imag'
        self.assertRaises(RuntimeError, lambda: hasattr(x, imag_key))
        self.assertTrue(hasattr(y, imag_key))
        keys.remove(imag_key)

        for key in keys:
            self.assertTrue(hasattr(x, key))


    def test_inplace_on_view_saved_output(self):
        # Test an in-place operation on a view in which the in-place op saves
        # its output. Previously, this created a reference cycle.
        dealloc = [0]

        class IncrementOnDelete(object):
            def __del__(self):
                dealloc[0] += 1

        def test():
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)

        test()
        self.assertEqual(dealloc[0], 1)

    def test_inplace_on_view_leaf_errors(self):
        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(RuntimeError,
                                    "a view of a leaf Variable that "
                                    "requires grad is being used in "
                                    "an in-place operation."):
            y.add_(1)

    def test_inplace_on_view_backward(self):
        # Issue #10532: Make sure that this does not raise RuntimeError.
        net = nn.Sequential(
            nn.InstanceNorm2d(2),
            nn.ReLU(True)
        )

        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        g, = torch.autograd.grad(net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape) , create_graph=True)
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # https://discuss.pytorch.org/t/freeing-buffer-strange-behavior/31955/8
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0., 0., True)
        prob_interpolated = torch.sigmoid(tmp2)

        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=inputs,
                                        grad_outputs=torch.ones(prob_interpolated.size()),
                                        create_graph=True, retain_graph=True)[0]

        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward0")

    def test_inplace_on_view_weak_grad_fn(self):
        # Issue 23502: Test that b's grad_fn is preserved.
        a = torch.arange(10.0, requires_grad=True)

        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        c = b.clone()
        del b
        gc.collect()

        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

    def test_out_variant_raises_when_inputs_require_grad(self):
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)

        # out=... functions don't support automatic differentiation currently
        self.assertRaisesRegex(RuntimeError, 'out=', lambda: torch.mul(a, b, out=x))

        # the inputs can require grad if we're in no_grad() mode
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)

        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        # we should throw an exception if the output requires grad
        self.assertRaisesRegex(RuntimeError, 'out=', lambda: torch.mul(a, b, out=x))

    def test_diagonal_derivative_requires_grad(self):
        # test that the backward requires grad
        # we do this is because diagonal_backward uses inplace
        # operations and gradgradcheck does not catch whether
        # they works as expected (it will succeed even if
        # the gradient has requires_grad == False
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.diagonal(a)**2
        c = b.sum()
        d, = torch.autograd.grad(c, a, retain_graph=True, create_graph=True)
        self.assertTrue(d.requires_grad)

    def test_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()  # Should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 0th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn('No forward pass information', str(w[0].message))

        inp = torch.rand(size, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 1th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out = MyFunc.apply(inp, inp, False)
                    out.backward()
            self.assertIn('MyFunc.apply', str(w[0].message))

    def test_nested_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                inp, = ctx.saved_tensors
                fail_0th = ctx.fail_0th
                g = gO.clone().expand(size)
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                return gI, None

        class MyFunc2(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1 * 2.0 + inp2

            @staticmethod
            def backward(ctx, gO):
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                # generate a nan
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return g1, g2, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        ginp, = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        gsum.backward()  # should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        ginp, = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 0th output."):
                with detect_anomaly():
                    gsum.backward()
        self.assertIn('No forward pass information', str(w[1].message))

        inp = torch.rand(size, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 1th output."):
                with detect_anomaly():
                    out = MyFunc.apply(inp, False)
                    ginp, = torch.autograd.grad(out, (inp,), create_graph=True)
                    gsum = ginp.sum()
                    gsum.backward()
        self.assertIn('MyFunc2.apply', str(w[1].message))
        self.assertIn('MyFunc.apply', str(w[2].message))

    def test_anomaly_grad_warnings(self):
        # PyTorch won't throw warnings if there is an error
        # but we'd want to at least see them in stderr

        class StdErrDiverter:
            def __enter__(self):
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig


        # if the warnings don't throw, they will be handled as regular warnings
        with self.assertRaisesRegex(RuntimeError,
                                    "one of the variables needed for gradient computation has been "
                                    "modified by an inplace operation"):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1 ** 2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 2)
        self.assertIn('Anomaly Detection has been enabled', str(w[0].message))
        self.assertIn('Error detected in PowBackward0', str(w[1].message))

        # if the warning throws, it will be printed to sys.stderr
        with self.assertRaisesRegex(RuntimeError,
                                    "one of the variables needed for gradient computation has been "
                                    "modified by an inplace operation"):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    warnings.simplefilter("error")
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1 ** 2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 1)
        self.assertIn('Anomaly Detection has been enabled', str(w[0].message))
        self.assertIn('Error detected in PowBackward0', s.captured)

    def test_anomaly_assign_parent_cleanup(self):
        # Test that python objects created are properly cleaned up when assign_parent is called
        import weakref

        def get_ref():
            # we use torch.exp here but any function that will construct a new node in its
            # backward call in grad mode will work
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()

            # ExpBackward calls mul, creating the MulBackward node when create_graph=True.
            # In anomaly mode, a PyObject referencing MulBackward's "parent" ExpBackward is added to
            # MulBackward's anomaly metadata dict, creating the following reference chain:
            #
            # grad -> MulBackward -> PyObject -> ExpBackward
            #
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            # We add a weak reference to a new Foo object, which we insert into ExpBackward's metadata dict
            #
            # (PyObject) -> ExpBackward -> dict -> *Foo*
            #            t ----^        WeakRef ---^
            #
            # We want to test that when grad goes out of scope at the end of this function that PyObject is destroyed
            # We can test this by seeing whether Foo is not kept alive once t is destroyed
            class Foo(object):
                pass
            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return t, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_nested_anomaly_printstack_cleanup(self):
        # Test if metadata dict PyObject is properly destroyed
        import weakref

        def get_ref():
            # This is similar to the construction in test_anomaly_assign_parent_cleanup:
            #
            # MyFuncBackward2 -> PyObject -> MyFuncBackward -> dict -> Foo
            #                               out ---^         WeakRef ---^
            #
            # We want to check that Foo is still properly destroyed even when MyFunc2Backward's
            # AnomalyMetadata calls printstack, which does some python object manipulation.
            #
            # You might be wondering why we still have to test_anomaly_assign_parent_cleanup,
            # since if PyObject is not destroyed here, wouldn't this test would detect that also?
            # The answer is that custom function's PyObject (THPFunction) actually only hold
            # a weak reference to the c++ node!
            class MyFunc(Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    x, = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO + float("NaN")

            inp = torch.rand(1, requires_grad=True)
            out = MyFunc.apply(inp)
            ginp, = torch.autograd.grad(out, (inp,), create_graph=True)

            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 0th output."):
                    with detect_anomaly():
                        ginp.backward()

            class Foo(object):
                pass
            my_obj = Foo()
            meta_dict = out.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return out, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    @skipIfNoLapack
    def test_eig_no_eigenvectors(self):
        A = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.eig(A, eigenvectors=False)
        with self.assertRaisesRegex(RuntimeError, 'is not differentiable'):
            torch.autograd.backward([w, v], [torch.ones_like(w), torch.ones_like(v)])

    @skipIfNoLapack
    def test_eig_complex_eigenvalues(self):
        A = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.eig(A, eigenvectors=True)
        with self.assertRaisesRegex(RuntimeError, 'does not support complex eigenvalues'):
            torch.autograd.backward([w, v], [torch.ones_like(w), torch.ones_like(v)])

    @skipIfNoLapack
    def test_symeig_no_eigenvectors(self):
        A = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.symeig(A, eigenvectors=False)
        with self.assertRaisesRegex(RuntimeError, 'is not differentiable'):
            torch.autograd.backward([w, v], [torch.ones_like(w), torch.ones_like(v)])

    def test_no_grad_copy(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.])

            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)

        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # non-contiguous grad should be copied
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        # test case that should trigger no copy for one of a,b
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

    def test_no_grad_copy_sparse(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # Create a sparse tensor with non-contigous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse.FloatTensor(ni, nv, torch.Size([10, 3]))
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for one of a,b
        emb_matrix = MyFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for i in range(10):
            loss.backward(retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        a.grad = b.grad = None
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # Verify we cloned both grads.
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for i in range(10):
            loss.backward(retain_graph=True)

    def test_gradcheck_single_input(self):
        def check(fast_mode):
            def f(inp):
                return inp.mul(5)

            gradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True), fast_mode=fast_mode)
            gradgradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True), fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_sparse_input(self):
        def check(fast_mode):
            def fn(sparse):
                return torch.sparse.sum(sparse)

            gradcheck(fn, torch.rand(10, dtype=torch.double).to_sparse().requires_grad_(True), check_sparse_nnz=True,
                      check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'gradcheck expects all tensor inputs are dense'):
                gradcheck(fn, torch.rand(10, dtype=torch.double).to_sparse().requires_grad_(True), check_sparse_nnz=False,
                          check_batched_grad=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.expectedFailure
    def test_gradcheck_sparse_csr_input(self):
        def check(fast_mode):
            def fn(sparse_csr):
                return torch.clone(sparse_csr).to_dense()

            # Fails because gradcheck can't work with sparse csr inputs yet
            gradcheck(fn, torch.rand(2, 2, dtype=torch.double).to_sparse_csr().requires_grad_(True), check_sparse_nnz=True,
                      check_batched_grad=False, fast_mode=fast_mode)

            with self.assertRaisesRegex(RuntimeError, 'gradcheck expects all tensor inputs are dense'):
                gradcheck(fn, torch.rand(2, 2, dtype=torch.double).to_sparse_csr().requires_grad_(True), check_sparse_nnz=False,
                          check_batched_grad=False, fast_mode=fast_mode)
        # check(fast_mode=True) # Segmentation fault
        check(fast_mode=False)

    def test_gradcheck_nondeterministic(self):
        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return NonDetFunc.apply(grad_out, ctx._jitter) * (1 + torch.rand_like(grad_out) * ctx._jitter), None

        def check(fast_mode):
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            gradcheck(lambda x: NonDetFunc.apply(x, 0.0), inp, check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Backward is not reentrant'):
                gradcheck(lambda x: NonDetFunc.apply(x, 1e-6), inp, check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Backward is not reentrant'):
                gradgradcheck(lambda x: NonDetFunc.apply(x, 1e-12), inp, check_batched_grad=False, fast_mode=fast_mode)
            gradcheck(lambda x: NonDetFunc.apply(x, 0.0), inp, nondet_tol=1e-5, check_batched_grad=False,
                      fast_mode=fast_mode)
            gradcheck(lambda x: NonDetFunc.apply(x, 1e-6), inp, nondet_tol=1e-5, check_batched_grad=False,
                      fast_mode=fast_mode)
            gradgradcheck(lambda x: NonDetFunc.apply(x, 1e-12), inp, nondet_tol=1e-5, check_batched_grad=False,
                          fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_validates_inputs(self):
        def check(fast_mode):
            # when inputs are not dense, but check_sparse_nnz is false
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'dense when check_sparse_nnz is set to False.'):
                gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=False, check_batched_grad=False,
                          fast_mode=fast_mode)
            self.assertFalse(gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=False,
                                       check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))

            # when none of the inputs require grad (always raises even if raise_exception=False)
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(ValueError, 'at least one input tensor to require gradient'):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

            # (warning) when inputs are not double precision
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(UserWarning, "Input #0 requires gradient and is not a double precision"):
                self.assertTrue(gradcheck(lambda x: x, (x,), atol=1e-1, fast_mode=fast_mode))

            # when layout is not mkldnn(aka has strides) and input has a dimension with stride 0. (always raises
            # even if raise_exception=False)
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))
            with self.assertRaisesRegex(RuntimeError, 'The 0th input has a dimension with stride 0'):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_gradcheck_validates_input_mkldnn(self):
        # when mkldnn inputs, forward mode testing is not allowed
        # Update tolerances below to make sure the gradient match even in single precision floats
        # Use the warning assert to hide the float32 warning
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(UserWarning, "Input #0 requires gradient and is not a double precision"):
            with self.assertRaisesRegex(ValueError, 'MKLDNN inputs are not support for forward AD gradcheck.'):
                gradcheck(lambda x: x.to_dense(), (x,), raise_exception=False, fast_mode=False, check_forward_ad=True,
                          atol=1e-1, rtol=1e-1)

        with self.assertWarnsRegex(UserWarning, "Input #0 requires gradient and is not a double precision"):
            with self.assertRaisesRegex(ValueError, 'MKLDNN inputs are not support for forward AD gradcheck.'):
                gradcheck(lambda x: x.to_dense(), (x,), raise_exception=False, fast_mode=True, check_forward_ad=True,
                          atol=1e-1, rtol=1e-1)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_gradcheck_test_outputs(self):
        def check(fast_mode):
            # when sparse outputs (always raise even if raise_exception=False)
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(ValueError, 'Sparse output is not supported at gradcheck yet'):
                gradcheck(lambda x: x, (x,), check_sparse_nnz=True, check_batched_grad=False, raise_exception=False,
                          fast_mode=fast_mode)

            # when mkldnn outputs (always raise even if raise_exception=False)
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'MKLDNN output is not supported at gradcheck yet'):
                gradcheck(lambda x: x.to_mkldnn(), (root,), check_batched_grad=False, raise_exception=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_no_differentiable_outputs(self):
        def check(fast_mode):
            # When none of the outputs are differentiable, but numerical gradient is not zero
            x = torch.ones((1,), requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'Numerical gradient for function expected to be zero'):
                gradcheck(lambda x: torch.tensor([x]), x)
            self.assertFalse(gradcheck(lambda x: torch.tensor([x]), x, raise_exception=False, fast_mode=fast_mode))

            # succeed when no outputs at all
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        def check(fast_mode):
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            # runtime error while compute batched grad (print big error)
            with self.assertRaisesRegex(RuntimeError, 'gradcheck or gradgradcheck failed while testing batched gradient'):
                gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=True, check_batched_grad=True, fast_mode=fast_mode)
            self.assertFalse(gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=True, check_batched_grad=True,
                                       raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_backward_mul_by_grad_output(self):
        # when grad_input is sparse and has incorrect sparse_dim/dense_dim
        def check(fast_mode):
            def fn(x):
                def hook(grad):
                    if grad is not None:
                        return grad.to_dense().to_sparse(1)
                    return grad
                y = x.clone()
                y.register_hook(hook)
                return y.to_dense()
            x = torch.ones((2, 2), dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'grad is sparse tensor, but has incorrect sparse_dim'):
                gradcheck(fn, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False,
                                       raise_exception=False, fast_mode=fast_mode))

            # when backward not multiplied by grad_output (non-sparse case)
            def fn2(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'backward not multiplied by grad_output'):
                gradcheck(fn2, (x,), atol=1e-1, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn2, (x,), atol=1e-1, raise_exception=False, fast_mode=fast_mode))

            # when backward not multiplied by grad_output (sparse case)
            def fn3(x):
                y = x.clone().to_dense()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'backward not multiplied by grad_output'):
                gradcheck(fn3, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn3, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False,
                                       raise_exception=False, fast_mode=fast_mode))

            # when layout of grad_input is not the same as input
            class Test(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, x):
                    return x.to_sparse()
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'grad is incorrect layout'):
                gradcheck(Test.apply, (x,), check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(Test.apply, (x,), check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_undefined_grad(self):
        def check(fast_mode):
            # when encounter runtime error while running backward
            def fn(x):
                def hook(x):
                    if x is None:
                        raise RuntimeError("x is undefined")
                y = x.clone()
                y.register_hook(hook)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertWarnsRegex(UserWarning, "Backwards compatibility: New undefined gradient support checking feature"):
                with self.assertRaisesRegex(RuntimeError, 'Expected backward function to handle undefined output grads'):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_jacobian_mismatch(self):
        def check(fast_mode):
            def fn(x):  # R -> R, C -> C
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))

            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, 'While considering the imaginary part of complex outputs only'):
                gradcheck(fn, (x_c,), fast_mode=False)
            self.assertFalse(gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False))

            def fn2(x):  # R -> C
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'While considering the imaginary part of complex outputs only'):
                gradcheck(fn2, (x,), fast_mode=False)
            self.assertFalse(gradcheck(fn2, (x,), raise_exception=False, fast_mode=False))

            def fn3(x):  # C -> R
                y = torch.real(x)
                y.register_hook(lambda x: x + 1e-2)
                return y
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn3, (x_c,), fast_mode=False)
            self.assertFalse(gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_dense_and_sparse_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x * y.coalesce().to_dense()
            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            b = torch.rand(2, 2, dtype=torch.double,).to_sparse().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, b), check_sparse_nnz=True, check_batched_grad=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_gradcheck_multiple_mkldnn_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x + y.to_dense()
            a = torch.rand(10, requires_grad=True)
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, b), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode))

            def fn2(x, y):
                return x.to_dense() + y.to_dense()
            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, c), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        def check(fast_mode):
            def fn(x):
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x
            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(AssertionError, 'return outputs with the same shape when inputs are perturbed'):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            def fn2(x):
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x
            with self.assertRaisesRegex(AssertionError, 'return outputs with the same dtype when inputs are perturbed'):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_complex_non_complex_outputs(self):
        def fn(x, y):
            z = torch.complex(x, y)
            return z, x + 1
        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        self.assertTrue(gradcheck(fn, (a, b)))

        def fn2(z):
            return z, torch.real(z)
        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        self.assertTrue(gradcheck(fn2, (c)))

    def test_gradcheck_get_numerical_jacobian(self):
        # get_numerical_jacobian is deprecated and no longer used internally by gradcheck
        from torch.autograd.gradcheck import get_numerical_jacobian

        def fn(inputs):
            # get_numerical_jacobian requires fn to take inputs as a tuple
            # and returns the jacobian wrt the first output
            x = inputs[0]
            y = inputs[1]
            return 2 * x + y, x + 2 * y
        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        with self.assertWarnsRegex(UserWarning, "get_numerical_jacobian was part of PyTorch's private API"):
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))

        with self.assertWarnsRegex(UserWarning, "get_numerical_jacobian was part of PyTorch's private API"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6, grad_out=2.0)

    def test_gradcheck_get_analytical_jacobian(self):
        from torch.autograd.gradcheck import get_analytical_jacobian

        def fn(x, y):
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        outputs = fn(a, b)
        with self.assertWarnsRegex(UserWarning, "get_analytical_jacobian was part of PyTorch's private API"):
            jacobians, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian((a, b), outputs[0])
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        self.assertTrue(reentrant)

        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return NonDetFunc.apply(grad_out, ctx._jitter) * (1 + torch.rand_like(grad_out) * ctx._jitter), None

        outputs = NonDetFunc.apply(a, 1e-6)
        with self.assertWarnsRegex(UserWarning, "get_analytical_jacobian was part of PyTorch's private API"):
            jacobians, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian((a,), outputs)
        self.assertFalse(reentrant)

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobians, _, _, _ = get_analytical_jacobian((a,), outputs, grad_out=2.0)

    def test_gradcheck_custom_error(self):
        from torch.autograd.gradcheck import GradcheckError

        def check(fast_mode):
            def fn(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(GradcheckError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))

            def fn2(x):
                raise RuntimeError("Not a GradcheckError!")
            # Checks that when raise_exception=False, non-GradcheckErrors are not caught by gradcheck
            with self.assertRaisesRegex(RuntimeError, "Not a GradcheckError!"):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_forward_ad(self):
        def fn(x, y):
            return x + y, y

        def bad_fn(x, y):
            # Hacky way to check if we're currently inside a forward ad level
            is_running_forward_ad = fwAD._current_level >= 0

            if is_running_forward_ad:
                y_p, y_d = fwAD.unpack_dual(y)
                y = fwAD.make_dual(y_p, y_d * 1.1)

            return x + y, y

        err_msg = "Jacobian computed with forward mode mismatch for output 0 with respect to input 1"

        for fast_mode in [True, False]:
            # Test for all inputs and outputs being real
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            def basic_mul(x):
                return torch.view_as_real(torch.resolve_conj(x * 1j))
            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)

            # Test for one input and one output being complex
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # Test for all inputs and outputs being complex
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

    def test_gradcheck_forward_ad_runs_with_no_requires_grad(self):
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        # This test checks that when the inputs are passed to the function they should not have
        # requires_grad=True even though they may have requires_grad=True when passed
        # to gradcheck
        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                if fwAD._current_level >= 0:
                    self.assertFalse(x.requires_grad)
                    self.assertFalse(y.requires_grad)
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                return x_t, y_t

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)

        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=False, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=False)

        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=False)

        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=True)

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=False)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=True)

    def test_gradcheck_forward_ad_respects_requires_grad(self):
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        jvp_count = [0]

        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                jvp_count[0] += 1
                return x_t, y_t

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=False, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=False)
        self.assertEqual(jvp_count[0], 2)  # (2) once per input
        jvp_count = [0]

        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=False)
        self.assertEqual(jvp_count[0], 6)  # (+4): (once with normal ZT (+1), once with efficient ZT (+1)) for each input (x2)
        jvp_count = [0]

        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=True)
        self.assertEqual(jvp_count[0], 12)  # (+6): (compute batch of 2 with vmap (+1), with a loop (+2)) for each input (x2)
        jvp_count = [0]

        # Repeat the previous test except we mark one input with requires_grad=False
        # NB: _test_undefined_forward_mode is only (+1), when function has single differentiable input, not (+2)!
        #     Otherwise, other counts are halved.
        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=False)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False,
                  check_batched_grad=False, check_batched_forward_grad=True)
        self.assertEqual(jvp_count[0], 5)  # 1 + 1 + 3

    def test_gradcheck_check_forward_or_backward_only(self):
        """Depending on settings for check_forward_ad and check_backward_ad, the
        correct codepaths should be reached (or not reached)
        """
        fwd_fail_err_msg = "FAIL FWD"
        bwd_fail_err_msg = "FAIL BWD"

        class UserFn(Function):
            @staticmethod
            def forward(ctx, foo, fwd_bad, bwd_bad):
                ctx.fwd_bad = fwd_bad
                ctx.bwd_bad = bwd_bad
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)
                else:
                    return 2 * gO, None, None

            @staticmethod
            def jvp(ctx, gI, _1, _2):
                if ctx.fwd_bad:
                    raise RuntimeError(fwd_fail_err_msg)
                else:
                    return 2 * gI

        for fast_mode in (True, False):
            for check_forward_ad in (True, False):
                for check_backward_ad in (True, False):
                    for fwd_bad in (True, False):
                        for bwd_bad in (True, False):
                            fwd_should_fail = fwd_bad and check_forward_ad
                            bwd_should_fail = bwd_bad and check_backward_ad

                            def run():
                                gradcheck(UserFn.apply, (x, fwd_bad, bwd_bad), check_forward_ad=check_forward_ad,
                                          check_backward_ad=check_backward_ad, check_undefined_grad=check_backward_ad,
                                          check_batched_grad=check_backward_ad, fast_mode=fast_mode)

                            x = torch.rand(2, dtype=torch.double, requires_grad=True)

                            if not check_forward_ad and not check_backward_ad:
                                with self.assertRaisesRegex(AssertionError, "Expected at least one of"):
                                    run()
                                continue

                            if not fwd_should_fail and not bwd_should_fail:
                                run()
                            else:
                                # If both fail, backward AD failure "hides" forward AD failure
                                if fwd_should_fail:
                                    fail_msg = fwd_fail_err_msg
                                if bwd_should_fail:
                                    fail_msg = bwd_fail_err_msg
                                with self.assertRaisesRegex(RuntimeError, fail_msg):
                                    run()

    def test_gradcheck_forward_ad_batched_grad(self):
        x = torch.rand(2, dtype=torch.double, requires_grad=True)

        # multiple inputs and outputs with non-tensors inputs
        def fn1(a: torch.Tensor, b: int):
            return a.clone(), a + 1
        gradcheck(fn1, (x, 1), check_forward_ad=True, check_backward_ad=False, check_batched_grad=False,
                  check_undefined_grad=False, check_batched_forward_grad=True)

        # unrelated inputs: tangent for c is None
        def fn2(a: torch.Tensor, c: torch.Tensor):
            return a.clone()
        gradcheck(fn2, (x, x.clone()), check_forward_ad=True, check_backward_ad=False, check_batched_grad=False,
                  check_undefined_grad=False, check_batched_forward_grad=True)

        class Fn(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                return gO * 2

            @staticmethod
            def jvp(ctx, gI):
                torch.randn_like(gI)
                return gI * 2

        msg = "vmap: We do not yet support calling random operations inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            gradcheck(Fn.apply, (x,), check_forward_ad=True, check_batched_forward_grad=True)

    def test_version_counter(self):
        x = torch.randn(1, 2)

        # In-place op bumps version
        x_saved_version = x._version
        x.add_(1).add_(1)
        self.assertTrue(x._version > x_saved_version)

        # Differentiable view shares version counter
        xz = x[:]
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

        # `x.data = y` preserves version counter of `x`
        x_saved_version = x._version
        x.data = torch.randn(2, 3)
        self.assertTrue(x._version == x_saved_version)
        x.add_(1)
        self.assertTrue(x._version > x_saved_version)
        # Make sure `x` is still using the same version counter it shares with `xz`
        self.assertTrue(x._version == xz._version)

        # In-place op on `xz` also updates version of `x`,
        # because they share the version counter
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while sparse tensor has impl
        # of type `SparseTensorImpl`.
        x = torch.randn(1, 2)
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_s

    def test_set_data_preserve_pyobj(self):
        a = torch.randn(1, 2)
        b = torch.randn(1, 2)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        code = """import torch
from torch.autograd import Function
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad
for shape in [(1,), ()]:
    v = torch.ones(shape, requires_grad=True)
    MyFunction.apply(v).backward()
"""
        s = TestCase.runWithPytorchAPIUsageStderr(code)
        # The autograd engine creates worker threads only when GPU devices are present.
        # So make sure that we do shutdown threads when we're testing cuda and make sure
        # that there is no thread to shutdown when we're not using cuda.
        if TEST_CUDA:
            self.assertRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")
        else:
            self.assertNotRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")

    @unittest.skipIf(IS_MACOS, "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941")
    def test_deep_reentrant(self):

        class DeepReentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x

        # Test stack overflow escape mechanism
        v = torch.tensor(2000.0, requires_grad=True)
        # This will cause stack overflow if reentrant calls are handled
        # in the same thread recursively
        DeepReentrant.apply(v).sum().backward()

        # Test stack overflow escape mechanism multiple times
        # to ensure reusing workers in the pool works fine
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()

    def test_reentrant_priority(self):
        order = []

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        class Reentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")


    @slowTest
    def test_checkpointing(self):
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp)
        )

        feat_combined = []
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r)
            feat_combined.append(feat_r)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

    @slowTest
    @parametrize("input_requires_grad", [True, False])
    def test_checkpointing_without_reentrant(self, input_requires_grad):
        """
        Basic test for checkpoint without reentrant autograd.
        """
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp)
        )

        # Run model with and without checkpointing and verify gradients are
        # equivalent, regardless of if inputs require grads or not.
        module_copy = deepcopy(module)

        feat_combined = []
        feat_combined_no_checkpoint = []
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = input_requires_grad
            data_r_copy = data_r.clone()
            feat_r = checkpoint(module, data_r, use_reentrant=False)
            feat_combined.append(feat_r)
            feat_r_no_checkpoint = module_copy(data_r)
            feat_combined_no_checkpoint.append(feat_r_no_checkpoint)


        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()
        mean_combined_no_checkpoint = torch.stack(feat_combined_no_checkpoint).mean()
        mean_combined_no_checkpoint.backward()

        for checkpoint_param, param in zip(module.parameters(), module_copy.parameters()):
            self.assertEqual(checkpoint_param.grad, param.grad)

    def test_checkpoint_valid_reset_on_error(self):
        a = torch.randn(2, 2, requires_grad=True)

        with self.assertRaisesRegex(Exception, "Checkpointing is not compatible with .grad()"):
            b = checkpoint(torch.exp, a).sum()
            torch.autograd.grad(b, (a,))

        c = checkpoint(torch.exp, a).sum()
        c.backward()

    @parametrize("use_reentrant", [True, False])
    def test_checkpointing_without_reentrant_detached_tensor(self, use_reentrant):
        class NoGradModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.lin2 = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                with torch.no_grad():
                    return self.lin2(self.linear(x))

        module = NoGradModule()

        err_ctx = (
            self.assertRaisesRegex(
                RuntimeError,
                "none of output has requires_grad=True"
            )
            if use_reentrant
            else contextlib.suppress()
        )

        a = torch.randn(2, 2, requires_grad=True)
        for _ in range(3):
            with err_ctx:
                # out does not require grad
                out = checkpoint(module, a, use_reentrant=use_reentrant)
                # Make loss require grad, otherwise we would run into
                # "element 0 of tensors does not require grad and does not have a grad_fn"
                out += a
                out.sum().backward()

    def test_checkpointing_without_reentrant_correct_grad(self):
        """
        Verifies that correct gradients are calculated for checkpoint
        without reentrant autograd, for both backward() and autograd.grad().
        """
        a = torch.randn(2, 2, requires_grad=True)

        b = torch.exp(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None
        c = checkpoint(torch.exp, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        a.grad = None
        d = checkpoint(torch.exp, a, use_reentrant=False).sum()
        d_grad, = torch.autograd.grad(d, (a,))

        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    def test_checkpointing_without_reentrant_dataparallel(self):
        """
        Verifies gradient correctness when checkpoint without reentrant autograd
        is used in conjunction with DataParallel.
        """
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                return self.linear(inp)

        a = torch.randn(2, 2, requires_grad=True)
        if torch.cuda.is_available():
            a = a.cuda()

        model = LinearModule()
        if torch.cuda.is_available():
            model = model.cuda()

        b = deepcopy(model)(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None

        module = torch.nn.DataParallel(deepcopy(model))
        c = checkpoint(module, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

