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

