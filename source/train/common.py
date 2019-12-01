import warnings

data_requirement = {}

def add_data_requirement(key, 
                         ndof, 
                         atomic = False, 
                         must = False, 
                         high_prec = False,
                         type_sel = None,
                         repeat = 1) :
    data_requirement[key] = {'ndof': ndof, 
                             'atomic': atomic,
                             'must': must, 
                             'high_prec': high_prec,
                             'type_sel': type_sel,
                             'repeat': repeat,
    }
    

class ClassArg () : 
    def __init__ (self) :
        self.arg_dict = {}
        self.alias_map = {}
    
    def add (self, 
             key,
             types_,
             alias = None,
             default = None, 
             must = False) :
        if type(types_) is not list :
            types = [types_]
        else :
            types = types_
        if alias is not None :
            if type(alias) is not list :
                alias_ = [alias]
            else:
                alias_ = alias
        else :
            alias_ = []

        self.arg_dict[key] = {'types' : types,
                              'alias' : alias_,
                              'value' : default, 
                              'must': must}
        for ii in alias_ :
            self.alias_map[ii] = key

        return self


    def _add_single(self, key, data) :
        vtype = type(data)
        if not(vtype in self.arg_dict[key]['types']) :
            # try the type convertion to the first listed type
            try :
                vv = (self.arg_dict[key]['types'][0])(data)
            except TypeError:
                raise TypeError ("cannot convert provided key \"%s\" to type %s " % (key, str(self.arg_dict[key]['types'][0])) )
        else :
            vv = data
        self.arg_dict[key]['value'] = vv

    
    def _check_must(self) :
        for kk in self.arg_dict:
            if self.arg_dict[kk]['must'] and self.arg_dict[kk]['value'] is None:
                raise RuntimeError('key \"%s\" must be provided' % kk)


    def parse(self, jdata) :
        for kk in jdata.keys() :
            if kk in self.arg_dict :
                key = kk
                self._add_single(key, jdata[kk])
            else:
                if kk in self.alias_map: 
                    key = self.alias_map[kk]
                    self._add_single(key, jdata[kk])
        self._check_must()
        return self.get_dict()

    def get_dict(self) :
        ret = {}
        for kk in self.arg_dict.keys() :
            ret[kk] = self.arg_dict[kk]['value']
        return ret

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json database must provide key " + key )
    else :
        return jdata[key]

def j_must_have_d (jdata, key, deprecated_key) :
    if not key in jdata.keys() :
        # raise RuntimeError ("json database must provide key " + key )
        for ii in deprecated_key :
            if ii in jdata.keys() :
                warnings.warn("the key \"%s\" is deprecated, please use \"%s\" instead" % (ii,key))
                return jdata[ii]
        raise RuntimeError ("json database must provide key " + key )        
    else :
        return jdata[key]

def j_have (jdata, key) :
    return key in jdata.keys() 

