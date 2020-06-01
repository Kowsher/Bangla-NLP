from pathlib import Path
script_location = Path(__file__).absolute().parent
yaml_loc = script_location / "iden.yaml"

with open(yaml_loc,'r',encoding = 'utf-8') as ide:
    iden = [word for line in ide for word in line.split()]


def n_gram(x, n):
    y = []
    idx = 0
    ln = len(x)
    cnt = 0
    temp = ""
    while idx<ln:
        
        if temp == "" and x[idx] in iden:
            idx = idx + 1
        if x[idx] == " ":
            temp = ""
            cnt = 0
        elif x[idx] in iden:
            temp = temp + x[idx]
        else:
            temp = temp + x[idx]
            cnt = cnt + 1
        if cnt ==n:
            if idx + 1 < ln:
                if x[idx+1] in iden:
                    temp = temp + x[idx+1]
            cnt = 0
            y.append(temp)
           
            temp = ""
            idx = idx + 1 - n
            
        
        idx = idx + 1
    return y    
    
 #=========================================