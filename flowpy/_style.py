from matplotlib import RcParams
import warnings
import re

PIPE = 1
CHANNEL = 2
BLAYER = 3


class StyleHandler():
    def __init__(self,flow_type,style_params=None):
        if style_params is None:
                self._style_params = get_default_style(flow_type)
        else:
            self._style_params = get_style_params(**style_params)

    def __getattr__(self,attr):
        return self._style_params[attr]

    def format_location(self,text):
        if text.count('=') == 0:
            return text
        split_text = text.split('=')
        floats = re.findall("\d+\.\d+|\d+|\-\d+\.\d+|\-\d+",split_text[-1])
        new_numbers = [float(x) for x in floats ]
        new_strs = [self.locationStyle(x) for x in new_numbers]
        for f, nf in zip(floats,new_strs):
            text = text.replace(f,nf)
        return text

    def create_label(self, text):
        text_list = list(text)
        for i in range(len(text_list)):
            text_list[i] = self.coord_mapping(text_list[i])
            text_list[i] = self.comp_mapping(text_list[i])

        return self.format_location("".join(text_list))  

def _validate_func_str2str(f):
    s = "stuff"
    try:
        ret = f(s)
        if not isinstance(ret,str):
            raise ValueError("Style parameters must return a str")
    except Exception as e:
        e.args[0] = "Validation raise exception: " + e.args[0]
        raise e

def _validate_func_g2str(f):
    s = 9.0
    try:
        ret = f(s)
        if not isinstance(ret,str):
            raise ValueError("Style parameters must return a str")
    except Exception as e:
        e.args[0] = "Validation raise exception: " + e.args[0]
        raise e
        
_param_keys = ['timeStyle', 'coordLabel', 'locationStyle', 'avgStyle']
_validators = [_validate_func_str2str]*4 + [_validate_func_g2str]
_validate_params = dict(zip(_param_keys,_validators))

def get_style_params(**params):
    for k in _param_keys:
        if k not in params:
            raise KeyError(f"{k} must be present in the style dictionary")

    if 'coord_mapping' not in _param_keys:
        params['coord_mapping'] = lambda x: r"%s"%x
        _validate_params['coord_mapping'] = _validate_func_str2str

    if 'comp_mapping' not in _param_keys:
        params['comp_mapping'] = lambda x: r"%s"%x
        _validate_params['comp_mapping'] = _validate_func_str2str        

    styleparams = RcParams()
    styleparams.validate = _validate_params
    dict.update(styleparams,params)
    return params

_defaults = {PIPE : None,
             CHANNEL : None,
             BLAYER : None}

_param_names = {PIPE : "PIPE",
                CHANNEL : "CHANNEL",
                BLAYER : "BLAYER"}
def set_default_style(flow_type,params):
    if flow_type not in [PIPE,CHANNEL,BLAYER]:
        raise ValueError("Invalid flow tyle")

    _defaults[flow_type] = get_style_params(**params)

def get_default_style(flow_type):
    if flow_type not in [PIPE,CHANNEL,BLAYER]:
        raise ValueError("Invalid flow tyle")

    if _defaults[flow_type] is None:
        raise KeyError("Default style has not been set for %s"%_param_names[flow_type])

    return _defaults[flow_type]

def update_default_style(flow_type,params):
    if flow_type not in [PIPE,CHANNEL,BLAYER]:
        raise ValueError("Invalid flow tyle")

    dict.update(_defaults,params)
    