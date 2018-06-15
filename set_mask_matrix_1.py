import numpy as np
import pandas as pd

def set_mask_matrix_1(h_matrix):
    parameters = {}
    row_num,column_num = h_matrix.shape
    vn_num = h_matrix.sum()
    cn_num = vn_num

    label_inv2c = []
    for col in range(column_num):
        vn2cnl = list(np.where(h_matrix[:,col]==1)[0])
        indd = 'v_%d'%(col)
        for l in vn2cnl:
            indd = indd + '_%d'%(l)
        label_inv2c.append(indd)

    label_c2v = []
    for row in range(row_num):
        cn2vnl = list(np.where(h_matrix[row,:]==1)[0])
        indd = 'c_%d'%(row)
        for c in cn2vnl:
            label_c2v.append(indd+'_%d'%(c))

    label_v2c = []
    for col in range(column_num):
        v2cl = list(np.where(h_matrix[:,col]==1)[0])
        indd = 'v_%d'%(col)
        for l in v2cl:
            label_v2c.append(indd+'_%d'%(l))

    mask1 = np.zeros((column_num,cn_num), dtype = np.float64)
    mask1 = pd.DataFrame(mask1, index=label_inv2c, columns=label_c2v)
    for ind in mask1.index:
        tmp = ind.split('_')
        vnode = tmp[1]
        v2cnode = tmp[2:]
        for col in mask1.columns:
            tmp = col.split('_')
            cnode = tmp[1]
            c2vnode = tmp[2]
            if cnode in v2cnode and vnode != c2vnode:
                mask1.loc[ind,col] = 1

    mask2 = np.zeros((cn_num,vn_num), dtype = np.float64)
    mask2 = pd.DataFrame(mask2, index=label_c2v, columns=label_v2c)
    for ind in mask2.index:
        tmp = ind.split('_')
        cnode = tmp[1]
        c2vnode = tmp[2]
        for col in mask2.columns:
            tmp = col.split('_')
            vnode = tmp[1]
            v2cnode = tmp[2]
            if c2vnode == vnode and v2cnode != cnode:
                mask2.loc[ind,col] = 1
    
    mask3 = np.zeros((vn_num,cn_num), dtype = np.float64)
    mask3 = pd.DataFrame(mask3, index=label_v2c, columns=label_c2v)
    for ind in mask3.index:
        tmp = ind.split('_')
        vnode = tmp[1]
        v2cnode = tmp[2]
        for col in mask3.columns:
            tmp = col.split('_')
            cnode = tmp[1]
            c2vnode = tmp[2]
            if v2cnode == cnode and vnode != c2vnode:
                mask3.loc[ind,col] = 1
    
    mask4 = np.zeros((cn_num,column_num), dtype = np.float64)
    mask4 = pd.DataFrame(mask4, index=label_c2v, columns=label_inv2c)
    for ind in mask4.index:
        tmp = ind.split('_')
        c2vnode = tmp[2]
        for col in mask4.columns:
            tmp = col.split('_')
            vnode = tmp[1]
            if vnode == c2vnode:
                mask4.loc[ind,col] = 1
    
    mask5 = np.zeros((column_num,vn_num), dtype = np.float64)
    mask5 = pd.DataFrame(mask5, index=label_inv2c, columns=label_v2c)
    for ind in mask5.index:
        tmp = ind.split('_')
        channelnode = tmp[1]
        for col in mask5.columns:
            tmp = col.split('_')
            vnode = tmp[1]
            if channelnode == vnode:
                mask5.loc[ind,col] = 1
    
    parameters['mask1'] = np.array(mask1)
    parameters['mask2'] = np.array(mask2)
    parameters['mask3'] = np.array(mask3)
    parameters['mask4'] = np.array(mask4)
    parameters['mask5'] = np.array(mask5)

    parameters['cn_num'] = cn_num
    parameters['vn_num'] = vn_num
    parameters['rows'] = row_num
    parameters['columns'] = column_num

    return parameters
 
