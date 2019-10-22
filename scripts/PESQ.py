# -*- coding: utf-8 -*-

import subprocess

                        
def analyzePESQ(p):
    out, err = p.communicate()
    start_index = out.index(b'=')+2
    end_index = out.index(b'\n',-1) # [start_index,end_index)
    
    score = out[start_index:end_index]
    score = float(score)
    
    return score
    
    
    
def getPESQ(tool_path,ref_file,deg_file):
    args = [tool_path+'pesq','+16000',ref_file,deg_file]                 
    p = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    score = analyzePESQ(p)
    return score
    
    
    
    
