
import re
import copy
import numpy as np
import scipy.io as sio

head_error = "\033[1;31;48m #ERROR \033[0m"

class FormatString:
    def __init__(self) -> None:
        pass

    def help(self):
        doc = """
\033[1;32;48m FormatString \033[0m
:Replace the formula <content> into its value
===============================================
\033[1;32;48m FOR \033[0m: <FOR(loop_configuration){loop_content}>
repeat generating the {loop_content} according to the {loop_configuration}
\t@loop_configuration: \"ii=NMIN:NMAX connect_char\"
\t@loop_content: any string you want to repeat
\t@example: <FOR(ii=1:3 \\n){A+B=ii;}> =>
\t\t A+B=1;\\nA+B=2;\\nA+B=3;

\033[1;32;48m IF \033[0m: <IF(judge_condition){content}>
generate the {content} according to the {judge_condition}
\t@judge_condition: any judgement condition
\t@content: any you want to generate if {judge_condition} is ture

\033[1;32;48m KEY \033[0m: <key>
use {key} as key to obtain its value in the dictoinary and use value replace it

\033[1;32;48m PRINT \033[0m: <PRINT(content)>
during generation procedure, print the {content} to screen

\033[1;32;48m ARR \033[0m: <ARR(key,idx1[,idx2][,idx3])
get the arrray {key} from dictionary,
and get the value according to the index {idx1}, {idx2}, and {idx3}

\033[1;32;48m DEF \033[0m: <DEF(key,value_express)>
update a key-value pair (i.e., {key} and {value_express}) into the dictinary
        """
        print(doc)

    def replace(self, dic, lines):
        """:Use the dic and command in line to generate a new line
        """
        # self.help()

        CODE_ENTER = 'CODE_ENTER'
        lines2 = []
        ii = 0
        n = len(lines)
        while(ii < n):
            line = lines[ii]
            ii += 1

            #empty line
            if len(line.split()) == 0:
                lines2.append(line)
                continue
            
            #block comment line
            if '/*' in line:
                lines2.append(line)
                while("*/" not in lines[ii-1]):
                    lines2.append(lines[ii])
                    ii += 1
                continue

            #print message
            if 'PRINT(' in line:
                a = re.search('PRINT(.*)', line)
                st, ed = a.span()
                line2 = line[st:ed-1]
                line2 = self.expressIdx2value(line2, dic)
                print(line2)
                continue

            #comment line
            if line.split()[0].startswith('//'):
                lines2.append(line)
                continue
            
            #continuation character
            if '//begin' in line:
                while('//end' not in lines[ii-1]):
                    line += lines[ii]
                    ii += 1
                line = line.replace('//begin', '')
                line = line.replace('//end', '')
                line = line.replace('\n', CODE_ENTER)
            
            
            #command line
            a = re.search('<[^=].*>', line)
            if a != None:
                st, ed = a.span()[0], a.span()[1]
                #test the '()' and '{}'
                n1 = np.sum([c == '(' for c in line[st:ed]])
                n2 = np.sum([c == ')' for c in line[st:ed]])
                n3 = np.sum([c == '{' for c in line[st:ed]])
                n4 = np.sum([c == '}' for c in line[st:ed]])
                if (n1 != n2) or (n3 != n4):
                    print(f'{head_error} : The number of brackets dismatch')
                    print('@line:', line.replace(CODE_ENTER, '\n'))
                    print('@(){\} n1, n2, n3, n4:', n1, n2, n3, n4)
                    exit()
                
                #s0: prefix
                #s1: <*>
                #s2: postfix
                s0 = line[:st]
                s1 = line[st+1:ed-1]
                s2 = line[ed:]

                #run command
                fmt = s1
                fmt = self.cmd_if(fmt, dic) #IF
                fmt = self.cmd_key(fmt, dic) #KEY
                fmt = self.cmd_for(fmt, dic) #FOR
                fmt = self.cmd_arr(fmt, dic) #ARR
                fmt, dic = self.cmd_def(fmt, dic) #DEF
                fmt = self.expressIdx2value(fmt, dic)

                line = s0 + fmt + s2
            
            line = line.replace(CODE_ENTER, '\n')
            lines2.append(line)

        return lines2

    ## FS: FORMAT_STRING
    #===================

    def express2value(self, fmt, dic):
        ''':eval the value of {fmt}
        fmt: the format string, such as 'a+b+1'
        dic: the dictory includes the value of key in {fmt}
            such as the dic={'a':2, 'b':3}, return value is 6
        '''
        # find the variable_name (string)
        a = re.findall('[a-zA-Z0-9_]+', fmt)
        #sort by length
        a = np.array(a)
        al = [len(ai) for ai in a]
        al = np.array(al)
        idx = np.argsort(-al)
        a = a[idx]
        # replace the variable_name(string) to variable_value(string)
        for key in a:
            if key in dic.keys():
                fmt = fmt.replace(key, str(dic[key]))
        # find unknown variables
        a = re.findall('[a-zA-Z]+', fmt)
        a2 = []
        for ai in a:
            if ai not in ['and','or','not', 'e']:
                a2.append(ai)
        if len(a2) == 0: 
            return eval(fmt)
        else: 
            return fmt

    def expressIdx2value(self, fmt, dic, ijk=None, vijk=None):
        ''':find '[variable_name]' and replace the {variable_name} to tis value according to dic
        '''
        #find '[*]' or '[*:*]'
        a = re.findall('\[[^[\]]+\]', fmt)
        # a = re.findall('\[[^[]+\]', fmt)
        # a = re.findall('\[[a-zA-Z0-9_+\-\*/ :\(\)]+\]', fmt)
        #dic
        dic2 = {}
        if ijk != None and vijk != None:
            dic2[ijk] = vijk
        dic2.update(dic)
        #sort by length
        a = np.array(a)
        al = [len(ai) for ai in a]
        al = np.array(al)
        idx = np.argsort(-al)
        a = a[idx]
        #replace
        for key in a:
            #'[*]' to *
            key2 = key.replace('[','').replace(']','')
            #'[*:*]' to * and *
            splt_key = re.findall('[+-]*:', key2)
            splt_key = splt_key[0] if len(splt_key) > 0 else ':'
            keys2 = key2.split(splt_key)
            p = True
            vlist = []
            for k in keys2:
                v = self.express2value(k, dic2)
                vlist.append(str(v))
                if type(v) == str:
                    p = False
            v = splt_key.join(vlist)
            if (not p) or (':' in v):
                fmt = fmt.replace(key, '['+v+']')
            else:
                fmt = fmt.replace(key, str(v))
        return fmt

    ## FSX_LOOP
    def get_loop_config(self, fmt, dic):
        ''': get the configuration of {cmd_for} command
        the syntax of configuration is:
        count=NMIN:NMAX str_concat
        @count: a notation which will increase from NMIN to NMAX
        the NMIN and NMAX whill be replace from notation to value according to {dic}
        '''
        pars = fmt.split() # count=NMIN:NMAX & str_concat
        sidx = pars[0]
        idx_pars = sidx.split('=') # count & NMIN:NMAX
        ijk = idx_pars[0] #count
        srange = idx_pars[1].split(':') # NMIN & NMAX
        while '' in srange:
            srange.remove('')
        if len(srange) == 2:
            smin, smax = srange
        else:
            print(f"{head_error} : The range is false for FOR configuration")
            print("@fun:", "get_loop_config")
            print("@fmt:", fmt)
            exit()
        # print(smin, smax)
        nmin = self.express2value(smin, dic)
        nmax = self.express2value(smax, dic)
        if str(nmin).isdigit() and str(nmax).isdigit():
            nmin = int(nmin)
            nmax = int(nmax)
        else:
            print(f"{head_error} : The NMIN and NMAX of FOR configuration can not be transfered to value:")
            print('@nmin, nmax:', nmin, nmax)
            print('@fmt:', fmt)
            print('@dic:', dic.keys())
            exit()
        if len(pars) == 1: str_concat = ""
        elif len(pars) == 2: str_concat = pars[1]
        return ijk, nmin, nmax, str_concat

    def get_bracket_position(self, line, brack='{', nbrack='}', st=0):
        """:find the position of brackets in the line
        """
        num_b = 0
        num_nb = 0 
        b_st = 0
        b_ed = 0
        for ii in range(st, len(line)):
            if line[ii] == brack:
                if num_b == 0:
                    b_st = ii
                num_b += 1
            if line[ii] == nbrack:
                num_nb += 1
            if (num_b > 0) and (num_nb == num_b):
                b_ed = ii
                break
        return b_st, b_ed

    def cmd_for(self, fmt, dic):
        """:use the 'FOR(loop_configuration){loop_content}' format to generate the repetitive verilog code
        @fmt: the format string to generate verilog code
        @dic: a dictory that includes the parameter in the {fmt} 
        @loop_configuration: is like "count=NMIN:NMAX str_concat"
        @loop_content: is like "A[count+1]=A[count];"
        """
        a = re.search('FOR\([^{}]+\)', fmt)
        # a = re.search('FOR[(0-9a-zA-Z=:,+\-\*/\\\ )]+', fmt)
        if a == None:
            fmt = fmt.replace('\\n', '\n')
            return fmt
        else:
            # find the head of loop format
            l = len(fmt)
            st, ed = a.span()[0], a.span()[1]
            sidx = fmt[st+4:ed-1] # '*' in 'FOR(*)'
            # get the loop information from the head
            ijk, nmin, nmax, str_concat = self.get_loop_config(sidx, dic)
            # find the string which need to be repeat generate
            stb, edb = self.get_bracket_position(fmt, '{', '}', ed)

            # s0: prefix
            # s1: loop setting
            # s2: {*}
            # s3: postfix
            s0 = fmt[:st]
            s1 = fmt[st:ed]
            s2 = fmt[stb+1:edb]
            s3 = fmt[edb+1:]

            # generate
            lineList = []
            dx = 1 if nmin <= nmax else -1
            dic2 = {}
            dic2.update(dic)
            for ii in range(nmin, nmax+dx, dx):
                dic2[ijk] = ii
                line = self.expressIdx2value(s2, dic2, ijk, ii)
                lineList.append(self.cmd_for(line, dic2))
            s = s0 + (str_concat.join(lineList)) + s3
            s = self.cmd_for(s, dic)
            s = s.replace('\\n', '\n')
            return s

    def cmd_if(self, fmt, dic):
        """:use the 'IF(p){line}' format to generate the verilog code
        @fmt: the format string to generate verilog code
        @dic: a dictory that includes the parameter in the {fmt} 
        @p: is judgment formula, and {line} is the content
        if {p} == True, return {line}; else return ''
        """
        # a = re.search('IF\([(0-9a-zA-Z_=+\-\*/\\><) ]+\)', fmt)
        a = re.search('IF\([^{}]+\)', fmt)
        if a != None:
            l = len(fmt)
            st, ed = a.span()[0], a.span()[1]
            sidx = fmt[st+3:ed-1] # 'IF(' and ')'
            stb, edb = self.get_bracket_position(fmt, '{', '}', ed)
            # s0: prefix
            # s1: judgment formula
            # s2: {*}
            # s3: postfix
            s0 = fmt[:st]
            s1 = fmt[st:ed]
            s2 = fmt[stb+1:edb]
            s3 = fmt[edb+1:]

            # print('s0',s0)
            # print('s1',s1)
            # print('s2',s2)
            # print('s3',s3)

            # generate
            p = self.express2value(sidx, dic)
            # print(sidx, p, s2)
            if p:
                s2 = self.expressIdx2value(s2, dic)
                fmt = s0 + s2 + s3
            else:
                fmt = s0 + s3
            return self.cmd_if(fmt, dic)
        else:
            return fmt

    def cmd_key(self, fmt, dic):
        """:replace the key string to the value string from dic
        @fmt: the string with <key_string> format
        @dic: is a key-value dictionary, containing key_string-value_string pair
        """
        a = re.search('[a-zA-Z0-9_]+', fmt)
        if a != None:
            l = len(fmt)
            st, ed = a.span()[0], a.span()[1]
            if st == 0 and ed == l:
                key = fmt
                if key in dic.keys():
                    fmt = str(dic[key])
        return fmt

    def cmd_arr(self, fmt, dic):
        """:replace the string to the value of the arry
        the format is ARR(arr, [idx1, idx2, idx3])
        """
        while (True):
            a = re.search('ARR\([^()]+\)', fmt)
            if a != None:
                l = len(fmt)
                st, ed = a.span()[0], a.span()[1]
                sidx = fmt[st+4:ed-1] # 'ARR(' and ')'
                # s0: prefix
                # s1: command
                # s2: postfix
                s0 = fmt[:st]
                s1 = fmt[st:ed]
                s2 = fmt[ed:]
                # print(s0, '#', s1, '#', s2, '\n')

                # generate
                pars = sidx.split(',')
                arr_name = pars[0]
                idxs = pars[1:]
                arr = dic[arr_name]
                # print(arr_name, '=>', arr.shape, '=>', idxs)
                if len(idxs) == 0:
                    v = "%d"%(arr)
                else:
                    d = arr
                    for idx in idxs:
                        d = d[int(idx)-1]
                    v = "%d"%(d)
                fmt = s0 + v + s2
            else:
                break
        return fmt

    def cmd_def(self, fmt, dic):
        """:add the key-value pair to the dictionary
        the format is DEF(key,value_express)
        """
        while (True):
            a = re.search('DEF\([^()]+\)', fmt)
            if a != None:
                l = len(fmt)
                st, ed = a.span()[0], a.span()[1]
                sidx = fmt[st+4:ed-1] # 'DEF(' and ')'
                # s0: prefix
                # s1: command
                # s2: postfix
                s0 = fmt[:st]
                s1 = fmt[st:ed]
                s2 = fmt[ed:]

                # generate
                pars = sidx.split(',')
                key = pars[0]
                value = pars[1]
                dic[key] = self.express2value(value, dic)
                fmt = s0 + s2
            else:
                break
        return fmt, dic


## OTHER
#=======

