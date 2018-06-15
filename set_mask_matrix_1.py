import numpy as np
import pandas as pd

# 根据校验节点建立掩码矩阵
# 传入对应码的校验矩阵，和待存储的字典
def set_mask_matrix_1(h_matrix):
    parameters = {}
    row_num,column_num = h_matrix.shape
    vn_num = h_matrix.sum()
    cn_num = vn_num

    # 这里存的标签是输入层的变量节点要连接的校验节点
    # 如，v_2_0_2标志变量节点0和校验节点0、2相连
    label_inv2c = []
    for col in range(column_num):
        vn2cnl = list(np.where(h_matrix[:,col]==1)[0])
        indd = 'v_%d'%(col)
        for l in vn2cnl:
            indd = indd + '_%d'%(l)
        label_inv2c.append(indd)
    # print(label_inv2c)

    # 这里存的标签是校验节点与变量节点的连接
    # c_1_0，校验节点1和变量节点0相连
    label_c2v = []
    for row in range(row_num):
        cn2vnl = list(np.where(h_matrix[row,:]==1)[0])
        indd = 'c_%d'%(row)
        for c in cn2vnl:
            label_c2v.append(indd+'_%d'%(c))
    # print(label_c2v)

    # 这里存的标签是变量节点与校验节点的连接
    # 如v_2_0表示变量节点2和校验节点0相连
    label_v2c = []
    for col in range(column_num):
        v2cl = list(np.where(h_matrix[:,col]==1)[0])
        indd = 'v_%d'%(col)
        for l in v2cl:
            label_v2c.append(indd+'_%d'%(l))
    # print(label_v2c)
# """
# 只需要上面三个标签列表来构造掩码矩阵
# 输入层到第一校验层的mask1，shape：(column_num,cn_num)
# 校验层到变量层的mask2，shape：(cn_num,vn_num)
# 变量层到校验层的mask3，shape：(vn_num,cn_num)
# 校验层到输出层的mask4，shape：(cn_num,column_num)
# 信道信息到变量层的mask5，shape：(column_num，vn_num)
# """
    ############################################################################
    #输入层到第一层cn
    mask1 = np.zeros((column_num,cn_num), dtype = np.float64)
    mask1 = pd.DataFrame(mask1, index=label_inv2c, columns=label_c2v)
    for ind in mask1.index:
        # 分割
        tmp = ind.split('_')
        # 当前变量节点号
        vnode = tmp[1]
        # 其连接的校验节点号
        v2cnode = tmp[2:]
        # 下面在mask1的各个列中查找
        for col in mask1.columns:
            tmp = col.split('_')
            # 当前校验节点号
            cnode = tmp[1]
            # 该校验节点要连接的变量节点号
            c2vnode = tmp[2]
            # 如果当前校验节点号在v2cnode列表中，并且，vnode不等于c2vnode
            # 则对应位置设为1
            if cnode in v2cnode and vnode != c2vnode:
                mask1.loc[ind,col] = 1
    # print(np.array(mask1))

    ############################################################################
    #cn层到vn层
    mask2 = np.zeros((cn_num,vn_num), dtype = np.float64)
    mask2 = pd.DataFrame(mask2, index=label_c2v, columns=label_v2c)
    for ind in mask2.index:
        # 分割
        tmp = ind.split('_')
        # 当前校验节点编号
        cnode = tmp[1]
        # 其连接的变量节点编号
        c2vnode = tmp[2]
        # 下面在mask2的各个列中查找：
        for col in mask2.columns:
            tmp = col.split('_')
            # 当前变量节点号
            vnode = tmp[1]
            # 该变量节点连接的校验节点号
            v2cnode = tmp[2]
            # 如果变量节点号等于校验节点要连接的，并且本校验节点不等于该变量节点要连接的
            # 则设为1
            if c2vnode == vnode and v2cnode != cnode:
                mask2.loc[ind,col] = 1
    # print(np.array(mask2))

    ############################################################################
    #vn层到cn层
    mask3 = np.zeros((vn_num,cn_num), dtype = np.float64)
    mask3 = pd.DataFrame(mask3, index=label_v2c, columns=label_c2v)
    for ind in mask3.index:
        # 分割
        tmp = ind.split('_')
        # 当前变量节点号
        vnode = tmp[1]
        # 其连接的校验节点号
        v2cnode = tmp[2]
        # 下面在mask3的各个列中查找
        for col in mask3.columns:
            tmp = col.split('_')
            # 当前校验节点号
            cnode = tmp[1]
            # 该校验节点连接的变量节点号
            c2vnode = tmp[2]
            # 如果当前变量节点所连接的校验节点号等于当前校验节点
            # 并且，当前校验节点连接的变量节点号不等于当前变量节点号，则设为1
            if v2cnode == cnode and vnode != c2vnode:
                mask3.loc[ind,col] = 1
    # print(np.array(mask3))

    ############################################################################
    #cn层到输出层
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
    # print(np.array(mask4))

    ############################################################################
    #mask5，用于扩展信道输入到vn的多个节点
    mask5 = np.zeros((column_num,vn_num), dtype = np.float64)
    mask5 = pd.DataFrame(mask5, index=label_inv2c, columns=label_v2c)
    for ind in mask5.index:
        tmp = ind.split('_')
        # 当前信道上变量节点号
        channelnode = tmp[1]
        for col in mask5.columns:
            tmp = col.split('_')
            # 当前变量层上变量节点号
            vnode = tmp[1]
            if channelnode == vnode:
                mask5.loc[ind,col] = 1
    # print(np.array(mask5))

    parameters['mask1'] = np.array(mask1)
    parameters['mask2'] = np.array(mask2)
    parameters['mask3'] = np.array(mask3)
    parameters['mask4'] = np.array(mask4)
    parameters['mask5'] = np.array(mask5)

    # parameters['mask1'] = np.ones((column_num,cn_num), dtype = np.float64)
    # parameters['mask2'] = np.ones((cn_num,vn_num), dtype = np.float64)
    # parameters['mask3'] = np.ones((vn_num,cn_num), dtype = np.float64)
    # parameters['mask4'] = np.ones((cn_num,column_num), dtype = np.float64)
    # parameters['mask5'] = np.ones((column_num,vn_num), dtype = np.float64)

    parameters['cn_num'] = cn_num
    parameters['vn_num'] = vn_num
    parameters['rows'] = row_num
    parameters['columns'] = column_num

    return parameters
    # return
