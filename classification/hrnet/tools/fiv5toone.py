import os
import shutil
import random
def return_key_val(path):
    key_val = {}
    for path_ in os.listdir(path):
        path_2 = os.path.join(path, path_)
        for path__ in os.listdir(path_2):
            _path_ = os.path.join(path, path_, path__)
            if  path__.split('_')[0]  not in  key_val.keys():
                key_val[path__.split('_')[0]] = []
            key_val[path__.split('_')[0]].append(_path_)
    return  key_val

def to_fold(file_name, save_path, fold):
    for tmp in file_name:
        source_file = tmp
        if 'yes' == source_file.split('\\')[1]:
            dirs = os.path.join(save_path, fold , 'yes')
        else:
            dirs = os.path.join(save_path, fold , 'no')
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        target_file = os.path.join(dirs , tmp.split('\\')[-1])
        # print(source_file)
        # print(target_file)
        shutil.copy(source_file, target_file)

def main(i, k, save_path, key_val):
    key = list(key_val.keys())
    random.seed(1234)
    random.shuffle(key)
    n = int(len(key)/k)
    val = [i for i in range((i-1)*n, (i)*n)]
    for _, name in enumerate(key_val.keys()):
        if _ in val:
            to_fold(key_val[name], save_path,'val' )
        else:
            to_fold(key_val[name], save_path,'train' )

def case_5_1(i, k):
    path = 'D:/code/医学/deal_result'
    save_path = 'D:/code/HRNet-Image-Classification/imagenet/images'
    try:
        shutil.rmtree(save_path)
    except:
        pass
    key_val = return_key_val(path)
    main(i, k, save_path, key_val)

def return_imgs(path):
    yes = []
    no = []
    for path_ in os.listdir(path):
        path_2 = os.path.join(path, path_)
        for path__ in os.listdir(path_2):
            _path_ = os.path.join(path, path_, path__)
            if 'yes' == _path_.split('\\')[1]:
                yes.append(_path_)
            if 'no' in _path_.split('\\')[1]:
                no.append(_path_)
    return [yes, no]

def imgs_to_fold(file_name, save_path, fold):

        source_file = file_name
        # print('tmp:',source_file)
        if 'yes' in source_file:
            dirs = os.path.join(save_path, fold , 'yes')
        else:
            dirs = os.path.join(save_path, fold , 'no')
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        target_file = os.path.join(dirs , source_file.split('\\')[-1])
        # print('source_file:',source_file)
        # print('target_file:',target_file)
        shutil.copy(source_file, target_file)

def imgs_main(i, k, save_path, key_val):
    key = key_val
    n = int(len(key)/k)
    val = [i for i in range((i-1)*n, (i)*n)]
    # print(key)
    for _, name in enumerate(key_val):
        if _ in val:
            imgs_to_fold(key_val[_], save_path,'val' )
        else:
            imgs_to_fold(key_val[_], save_path,'train' )

def imgs_5_1(i, k):
    path = 'D:/code/医学/deal_result'
    save_path = 'D:/code/HRNet-Image-Classification/imagenet/images'
    try:
        shutil.rmtree(save_path)
    except:
        pass
    key_val = return_imgs(path)
    random.seed(1234)
    random.shuffle(key_val[0])
    random.shuffle(key_val[1])
    imgs_main(i, k,save_path, key_val[0])
    imgs_main(i, k,save_path, key_val[1])

# imgs_5_1()
# case_5_1()