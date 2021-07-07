import os 

def mk_dir(directory):
    # 폴더 없으면 폴더 생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod