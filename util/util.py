import os.path

def file_by_type(file_list, ext):
    """
    files_list -- list of files (e.g. from Ruffus)
    ext -- file type to match (e.g. '.txt')
    """
    filtered = [fname for fname in file_list if fname.endswith(ext)]
    assert len(filtered) == 1, "Expect unique match"
    return filtered[0]

def path_to_pypath(pyfile):
    """
    pyfile -- relative path to a python module (e.g. 'dir/module.py')
    returns -- python module (e.g. 'dir.module')
    """
    # TODO: use builtin importlib to perform the conversion between paths and module names
    # http://stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python
    root, ext = os.path.splitext(os.path.basename(pyfile))
    assert ext == '.py' # must load py file
    assert '.' not in root # not dots in file name (other than .py extention)
    pth = os.path.normpath(os.path.dirname(os.path.relpath(pyfile)))
    if pth == '.':
        # py file is in this directory
        return root
    assert '.' not in pth # no paths outside of this directory, or dots in directory names
    dirs = pth.split(os.sep)
    pypth = ".".join(dirs) + "." + root
    return pypth
