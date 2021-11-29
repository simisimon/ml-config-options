from pprint import pprint

def get_source_code(source_file):
    with open(source_file) as f:
        code = f.read()
    return code

def get_import_lines(code, ml_lib):
    import_lines = [line for line in code.split('\n') if line.startswith("from") or line.startswith("import")]
    import_lines = [line for line in import_lines if ml_lib in line]
    return import_lines

def get_search_words(ml_lib_import_lines, ml_lib):
    search_words = []
    import_lines_list = modify_import_lines(ml_lib_import_lines, "import")
    import_lines_list = [s for s in import_lines_list if s.startswith(ml_lib)] #removing irrelevant libraries

    for s in import_lines_list:
        if "as" in s:
            search_words.append(s.strip().split("as")[-1])
        else:
            search_words.append(s)

    from_lines_list = modify_import_lines(ml_lib_import_lines, "from")
    for s in from_lines_list:
        if "as" in s:
            search_words.append(s.strip().split("as")[-1])
        elif "import" in s:
            search_words.append(s.strip().split("import")[-1])
        else:
            search_words.append(s)

    return search_words

def modify_import_lines(ml_lib_import_lines, initial_word):
    lines_list = [line for line in ml_lib_import_lines if line.startswith(initial_word)] #all lines starting with import/from and containing library
    lines_list = [s.replace(" ", "") for s in lines_list]  # removing blank spaces
    lines_list = [s.replace(initial_word, "") for s in lines_list]  # removing import/from keyword
    lines_list = [s.split(",") for s in lines_list]  # seperating the libraries into nested list
    lines_list = [s for sublist in lines_list for s in sublist]  # flatten the nested list

    return lines_list

def reduce_code(code):
    code = [line for line in code.split('\n') if not (line.startswith("from") or line.startswith("import") or line.startswith("#") or line == "")]
    return code

def get_relevant_lines(code, search_words):
    lines = []
    for line in code:
        for search_word in search_words:
            if search_word in line:
                lines.append(line)

    return lines


source_file = "test_projects/sklearn_lin_reg.py"
ml_lib = "sklearn"
code = get_source_code(source_file)
import_lines = get_import_lines(code, ml_lib)
search_words = get_search_words(import_lines, ml_lib)
pprint(search_words)
code = reduce_code(code)
lines = get_relevant_lines(code, search_words)
pprint(lines)


