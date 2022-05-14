
import os
import numpy as np
import json
import struct


class FioHead():
    r""" output with color
    """
    def __init__(self):
        pass

    def info(msg='#INFO'):
        return '\033[1;32;48m #INFO \033[0m'

    def warning(msg='#WARNING'):
        return '\033[1;33;48m #WARNING \033[0m'

    def error(msg='#ERROR'):
        return '\033[1;31;48m #ERROR \033[0m'


class Fio:
    """ basic class for FIO
    """
    def __init__(self):
        pass

    def exits(self, file_name=''):
        if file_name == '':
            return True
        return os.path.exists(file_name)

    def mkdir(self, path_name=''):
        if not self.exits(path_name):
            # os.mkdir(path_name)
            os.makedirs(path_name)

    def create_file_path(self, file_name=''):
        pars = file_name.split('/')
        if len(pars) > 0:
            path_name = '/'.join(pars[:-1])
            self.mkdir(path_name)

    def is_path(self, path):
        return self.exits(path) and os.path.isdir(path)

    def is_file(self, file_name):
        return self.exits(file_name) and os.path.isfile(file_name)

    def get_file_list(self, path) -> list:
        if self.is_file(path):
            return []
        if self.is_path:
            listdir = os.listdir(path)
            file_lst = []
            for name in listdir:
                if self.is_file(path + '/' + name):
                    file_lst.append(path + '/' + name)
                else:
                    file_lst_ = self.get_file_list(path + '/' + name)
                    file_lst.extend(file_lst_)
            return file_lst
        return []


class FioDic:
    r""": input and output for dict class data
    the file can be .json or .npy file containing a dictionary
    """
    def __init__(self) -> None:
        pass

    def load(self, file_name='', default_value={}):
        if file_name.endswith('.json'):
            return FioJsonDic().load(file_name, default_value)
        elif file_name.endswith('.npy'):
            return FioNpyDic().load(file_name, default_value)
        else:
            return FioNpyDic().load(file_name, default_value)

    def save(self, file_name='', dic={}):
        if file_name.endswith('.json'):
            FioJsonDic().save(file_name, dic)
        elif file_name.endswith('.npy'):
            FioNpyDic().save(file_name, dic)
        else:
            FioNpyDic().save(file_name, dic)

    def get(self, jdata, key, default_value):
        if key in jdata.keys():
            return jdata[key]
        else:
            return default_value

    def update(self, jdata, jdata_o):
        """
        jdata: new jdata
        jdata_o: origin jdata
        """
        for key in jdata.keys():
            if key in jdata_o.keys():
                if isinstance(jdata_o[key], dict):
                    jdata_o[key] = self.update(jdata[key], jdata_o[key])
                else:
                    jdata_o[key] = jdata[key]
        return jdata_o


class FioNpyDic:
    r""": input and output for .npy file containing dictionary
    """
    def __init__(self):
        pass

    def load(self, file_name='', default_value={}):
        if Fio().exits(file_name):
            head = FioHead().info()
            print(f"{head}: load {file_name}")
            dat = np.load(file_name, allow_pickle=True)[0]
            return dat
        else:
            head = FioHead.warning()
            print(f"{head}: can not find {file_name}")
            return default_value

    def save(self, file_name='', dic={}):
        Fio().create_file_path(file_name)
        np.save(file_name, [dic])


class FioJsonDic:
    r""": input and output for .json file containing dictionary
    """
    def __init__(self):
        pass

    def load(self, file_name='', default_value={}):
        if Fio().exits(file_name):
            head = FioHead().info()
            print(f"{head}: load {file_name}")
            with open(file_name, 'r') as fr:
                jdata = fr.read()
            dat = json.loads(jdata)
            return dat
        else:
            head = FioHead().warning()
            print(f"{head}: can not find {file_name}")
            return default_value

    def save(self, file_name='', dic={}):
        head = FioHead().info()
        print(f"{head}: write jdata to {file_name}")
        Fio().create_file_path(file_name)
        with open(file_name, 'w') as fw:
            json.dump(dic, fw, indent=4)


class FioBin():
    r""": input and output for binary file
    """
    def __init__(self):
        pass

    def load(self, file_name='', default_value=''):
        if Fio().exits(file_name):
            head = FioHead().info()
            print(f"{head}: load {file_name}")
            dat = ""
            with open(file_name, 'rb') as fr:
                dat = fr.read()
            return dat
        else:
            head = FioHead().warning()
            print(f"{head}: can not find {file_name}")
            return default_value

    def save(self, file_name: str = '', data: str = ''):
        head = FioHead().info()
        print(f"{head}: write binary to {file_name}")
        Fio().create_file_path(file_name)
        with open(file_name, 'wb') as fp:
            for si in data:
                # one byte consists of two hex chars
                for ii in range(len(si) // 2):
                    v = int(si[2 * ii: 2 * (ii + 1)], 16)
                    v = struct.pack('B', v)
                    fp.write(v)


class FioTxt():
    r""": input and output for .txt file with string
    """
    def __init__(self):
        pass

    def load(self, file_name='', default_value=[]):
        if Fio().exits(file_name):
            head = FioHead().info()
            print(f"{head}: load {file_name}")
            with open(file_name, 'r', encoding='utf-8') as fr:
                dat = fr.readlines()
            dat = [d.replace('\n', '') for d in dat]
            return dat
        else:
            head = FioHead().warning()
            print(f"{head}: can not find {file_name}")
            return default_value

    def save(self, file_name: str = '', data: list = []):
        head = FioHead().info()
        print(f"{head}: write string to txt file {file_name}")
        Fio().create_file_path(file_name)

        if isinstance(data, str):
            data = [data]
        data = [d + '\n' for d in data]
        open(file_name, 'w').writelines(data)


class FioArrInt():
    r""": input and output for .txt file contianing a integer array
    """
    def __init__(self):
        pass

    def load(self, file_name='', default_value=[], nbit=10):
        if Fio().exits(file_name):
            head = FioHead().info()
            print(f"{head}: load {file_name}")
            dat = np.loadtxt(file_name, dtype=np.int)
            return dat
        else:
            head = FioHead().warning()
            print(f"{head}: can not find {file_name}")
            return default_value

    def save(self, file_name: str = '', data: list = [], nbit=10):
        # head = FioHead().info()
        # print(f"{head}: write string to txt file {file_name}")
        Fio().create_file_path(file_name)

        rng = int(2 ** (nbit + 1))
        nstr = len(str(rng))  # the length of string for display as decimal number
        if len(data.shape) <= 2:
            np.savetxt(file_name, data, fmt=f'%{nstr}d')
        elif len(data.shape) == 3:
            np.savetxt(file_name, data[0], fmt=f'%{nstr}d')
            with open(file_name, 'a') as fw:
                n = data.shape[0]
                for ii in range(1, n):
                    np.savetxt(fw, data[ii], fmt=f'%{nstr}d')
        else:
            pass
