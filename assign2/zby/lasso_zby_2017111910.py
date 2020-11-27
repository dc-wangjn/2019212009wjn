# Zhangboyang 2017111910

# 加载包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

# 导入数据
x_data, y_data = load_boston().data, load_boston().target

# 标准化
x_mean, x_std, y_mean, y_std = np.mean(x_data, axis=0), np.std(x_data, axis=0), np.mean(y_data, axis=0), np.std(y_data, axis=0)
x_standardization = np.hstack((np.ones((np.shape(x_data)[0],1)),(x_data-x_mean)/x_std))
y_standardization = (y_data-y_mean)/y_std

# 设定参数
s_number_para = 61
kfold_para = 5
mu_para = np.array([1e-6, 1e-5, 1e-4])
epsilon_para = np.array([1e-6, 1e-5, 1e-4])
convergence = np.array([0, 1, 2])
temp = np.vstack((np.array(range(0,s_number_para)),10**(-4+0.15*np.array(range(0,s_number_para))))).T
result_col_number = np.shape(x_standardization)[1]+7
result_1 = np.hstack((temp,np.zeros((s_number_para,3*result_col_number))))
result_2 = np.hstack((temp,np.zeros((s_number_para,9*result_col_number))))

# lasso函数
def zby_2017111910_hw2_lasso(y_in,x_in,kfold_in,mu_in,epsilon_in,convergence_in,lambda_in):
    # 定义返回参数
    mse_out = 0
    variance_out = 0
    biasq_out = 0
    iteration_out = 0
    
    # 创建变量
    dimension = x_in.shape[1]
    beta_eye = np.eye(dimension)*1e-4
    beta_tmp = np.zeros(dimension) 
    beta_1 = np.ones(dimension)
    beta_2 = np.zeros(dimension)
    
    # k折交叉法循环
    kf = KFold(n_splits=kfold_in)
    for train_index, test_index in kf.split(x_in):
        x_test = x_in[test_index]
        x_train = x_in[train_index]
        y_test = y_in[test_index]
        y_train = y_in[train_index]
        beta_1[:] = 1
        beta_2[:] = 0
        iteration_out = 0
        epsilon_cal=epsilon_in+1
        while iteration_out<= 15000 and epsilon_cal>=epsilon_in :
            beta_1 = beta_2+0
            iteration_out = iteration_out+1
            y_tmp = np.sum((y_train.reshape((y_train.shape[0],1))-np.dot(x_train, (beta_1+beta_eye).T) )**2,axis=0)
            beta_tmp = beta_1-mu_in*1e4*(y_tmp-sum((y_train-np.dot(x_train ,beta_1))**2))
            tmp_1 = (1-lambda_in*mu_in/abs(beta_tmp))
            beta_2 = beta_tmp*(tmp_1>0)*tmp_1
            epsilon_cal=np.linalg.norm(beta_2-beta_1) if convergence_in==0 else(
                np.linalg.norm(beta_2-beta_1)/(np.linalg.norm(beta_1)+1e-15) if convergence_in==1 else
                    np.linalg.norm(beta_2-beta_1)/(np.linalg.norm(beta_1)+epsilon_in)) 
            
        mse_out = mse_out + sum((y_test-np.dot(x_test,beta_1))**2)/y_in.shape[0]
        variance_out = variance_out + sum((np.mean(np.dot(x_test,beta_1))-np.dot(x_test,beta_1))**2)/y_in.shape[0]
        biasq_out = biasq_out + sum((y_test-np.mean(np.dot(x_test,beta_1)))**2)/y_in.shape[0]
    
    # 用所有数据训练一个beta出来作为最终的估计值
    beta_1[:] = 1
    beta_2[:] = 0
    iteration_out = 0
    epsilon_cal=epsilon_in+1
    
    while iteration_out<= 15000 and epsilon_cal>=epsilon_in :
        beta_1 = beta_2+0
        iteration_out = iteration_out+1 
        y_tmp = np.sum((y_train.reshape((y_train.shape[0],1))-np.dot(x_train, (beta_1+beta_eye).T) )**2,axis=0)
        beta_tmp = beta_1-mu_in*1e4*(y_tmp-sum((y_train-np.dot(x_train ,beta_1))**2))
        tmp_1 = (1-lambda_in*mu_in/abs(beta_tmp))
        beta_2 = beta_tmp*(tmp_1>0)*tmp_1
        epsilon_cal=np.linalg.norm(beta_2-beta_1) if convergence_in==0 else(
            np.linalg.norm(beta_2-beta_1)/(np.linalg.norm(beta_1)+1e-15) if convergence_in==1 else
                np.linalg.norm(beta_2-beta_1)/(np.linalg.norm(beta_1)+epsilon_in)) 
        
    return np.hstack((mu_in,epsilon_in,convergence_in,iteration_out,mse_out,variance_out,biasq_out,beta_2))

# 先跑一组数据看看不同的收敛方式带来的影响
for i in result_1[:,0]:
    result_1[int(i),range(2,result_1.shape[1])] = np.hstack((zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[1],epsilon_para[1],convergence[0],result_1[int(i),1]),  
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[1],epsilon_para[1],convergence[1],result_1[int(i),1]),
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[1],epsilon_para[1],convergence[2],result_1[int(i),1])))
    
# 画图
# 三种收敛方式下的迭代次数图
plt.title('Numbers of Iterations with 3 Types', fontsize=20)  
plt.plot(result_1[:,0], result_1[:,5], label='Absolute Convergence')
plt.plot(result_1[:,0], result_1[:,5+result_col_number],linestyle='--' , label='Relative Convergence')
plt.plot(result_1[:,0], result_1[:,5+result_col_number*2],linestyle=':', label='Adjusted Relative Convergence')
plt.xlabel('s, ($λ=10^{(-4+0.15s)}$)', fontsize=20)
plt.ylabel('Number of Iterations', fontsize=15)
plt.legend(bbox_to_anchor=(0.6, -0.2))
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# 三种收敛方式下的MSE图1
plt.title('Mean Squared Error with 3 Types', fontsize=20)  
plt.plot(result_1[:,0], result_1[:,6], label='Absolute Convergence')
plt.plot(result_1[:,0], result_1[:,6+result_col_number],linestyle='--' , label='Relative Convergence')
plt.plot(result_1[:,0], result_1[:,6+result_col_number*2],linestyle=':', label='Adjusted Relative Convergence')
plt.scatter(np.argmin(result_1[:,6]), min(result_1[:,6]), label='min value', marker = 'o', s=200)
plt.xlabel('s, ($λ=10^{(-4+0.15s)}$)', fontsize=20)
plt.ylabel('Mean Squared Error', fontsize=15)
plt.legend(bbox_to_anchor=(0.6, -0.2))
plt.text(np.argmin(result_1[:,6]), min(result_1[:,6])+0.05, '%.4f' % min(result_1[:,6]), ha='center', va= 'bottom',fontsize=15) 
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

np.argmin(result_1[:,6])
result_1[np.argmin(result_1[:,6]),1]
np.argmax(result_1[:,6])
result_1[np.argmax(result_1[:,6]),1]

# 三种收敛方式下的MSE图2（相对误差最大时）
print(np.argmax(abs((result_1[:,6+result_col_number]-result_1[:,6])/result_1[:,6])),max(abs((result_1[:,6+result_col_number]-result_1[:,6])/result_1[:,6])))
max_rela = np.argmax(abs((result_1[:,6+result_col_number]-result_1[:,6])/result_1[:,6]))

plt.title('Mean Squared Error with 3 Types', fontsize=20)  
plt.scatter(result_1[max_rela,0], result_1[max_rela,6], label='Absolute Convergence', marker = 'o', s=200)
plt.scatter(result_1[max_rela,0], result_1[max_rela,6+result_col_number], label='Relative Convergence', marker = 'o', s=200)
plt.scatter(result_1[max_rela,0], result_1[max_rela,6+result_col_number*2], label='Adjusted Relative Convergence', marker = 'o', s=100)
plt.xlabel('s, ($λ=10^{(-4+0.15s)}$)', fontsize=20)
plt.ylabel('Mean Squared Error', fontsize=15)
plt.legend(bbox_to_anchor=(0.6, -0.2))
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# 系数估计值
plt.plot(result_1[:,0], result_1[:,9], label='INTERCEPT')
plt.plot(result_1[:,0], result_1[:,10], label='CRIM')
plt.plot(result_1[:,0], result_1[:,11], label='ZN')
plt.plot(result_1[:,0], result_1[:,12], label='INDUS')
plt.plot(result_1[:,0], result_1[:,13], label='CHAS')
plt.plot(result_1[:,0], result_1[:,14], label='NOX')
plt.plot(result_1[:,0], result_1[:,15], label='RM')
plt.plot(result_1[:,0], result_1[:,16], label='AGE')
plt.plot(result_1[:,0], result_1[:,17], label='DIS')
plt.plot(result_1[:,0], result_1[:,18], label='RAD')
plt.plot(result_1[:,0], result_1[:,19], label='TAX')
plt.plot(result_1[:,0], result_1[:,20], label='PTRATIO')
plt.plot(result_1[:,0], result_1[:,21], label='B')
plt.plot(result_1[:,0], result_1[:,22], label='LSTAT')
plt.legend(bbox_to_anchor=(1, 1), fontsize=15)
plt.xlabel('s, ($λ=10^{(-4+0.15s)}$)', fontsize=20)
plt.ylabel('Standardized Coefficients', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# 提取系数
re_min = np.argmin(result_1[:,6])
print(result_1[re_min,range(0,9)])
print(result_1[re_min,range(9,23)])
beta_result = result_1[re_min,range(9,23)]+0
1-sum((y_standardization-np.dot(x_standardization,beta_result))**2)/sum((y_standardization)**2)

# 挑选变量
for i in range(0,beta_result.shape[0]):
    if abs(beta_result[i])<=0.1:
        beta_result[i] = 0
1-sum((y_standardization-np.dot(x_standardization,beta_result))**2)/sum((y_standardization)**2)
print(beta_result)

# 相关系数矩阵
print(np.corrcoef(x_standardization[:,[1,5,6,8,9,11,13]].T))

# 挑选变量
for i in range(0,beta_result.shape[0]):
    if abs(beta_result[i])<=0.12:
        beta_result[i] = 0
1-sum((y_standardization-np.dot(x_standardization,beta_result))**2)/sum((y_standardization)**2)
print(beta_result)

# 还原参数
beta_result_2 = result_1[re_min,range(9,23)]+0
beta_result_2[0] = (beta_result[0]-sum(beta_result[range(1,beta_result_2.shape[0])]*x_mean/x_std))*y_std+y_mean
beta_result_2[range(1,beta_result_2.shape[0])] = beta_result[range(1,beta_result_2.shape[0])]*y_std/x_std
print(beta_result_2)
1-sum((y_standardization-np.dot(x_standardization,beta_result))**2)/sum((y_standardization)**2)

# 再跑一组数据看看不同的ε、μ带来的影响
for i in result_2[:,0]:
    result_2[int(i),range(2,result_2.shape[1])] = np.hstack((zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[0],epsilon_para[0],convergence[0],result_1[int(i),1]),  
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[0],epsilon_para[1],convergence[0],result_1[int(i),1]),
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[0],epsilon_para[2],convergence[0],result_1[int(i),1]),
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[1],epsilon_para[0],convergence[0],result_1[int(i),1]),  
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[1],epsilon_para[1],convergence[0],result_1[int(i),1]),
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[1],epsilon_para[2],convergence[0],result_1[int(i),1]),
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[2],epsilon_para[0],convergence[0],result_1[int(i),1]),  
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[2],epsilon_para[1],convergence[0],result_1[int(i),1]),
                                                             zby_2017111910_hw2_lasso(y_standardization,x_standardization,kfold_para,mu_para[2],epsilon_para[2],convergence[0],result_1[int(i),1])))
    
# 九种参数组合的迭代次数图
plt.title('Numbers of Iterations', fontsize=20)  
plt.plot(result_2[:,0], result_2[:,5], label='μ=1e-6,ε=1e-6')
plt.plot(result_2[:,0], result_2[:,5+result_col_number],linestyle='--' , label='μ=1e-6,ε=1e-5')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*2],linestyle=':', label='μ=1e-6,ε=1e-4')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*3], label='μ=1e-5,ε=1e-6')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*4],linestyle='--' , label='μ=1e-5,ε=1e-5')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*5],linestyle=':', label='μ=1e-5,ε=1e-4')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*6], label='μ=1e-4,ε=1e-6')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*7],linestyle='--' , label='μ=1e-4,ε=1e-5')
plt.plot(result_2[:,0], result_2[:,5+result_col_number*8],linestyle=':', label='μ=1e-4,ε=1e-4')
plt.xlabel('s, ($λ=10^{(-4+0.15s)}$)', fontsize=20)
plt.ylabel('Number of Iterations', fontsize=15)
plt.legend(bbox_to_anchor=(1, 1), fontsize=15)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# 九种参数组合的mse图
plt.title('Mean Squared Error', fontsize=20)  
plt.plot(result_2[:,0], result_2[:,6], label='μ=1e-6,ε=1e-6')
plt.plot(result_2[:,0], result_2[:,6+result_col_number],linestyle='--' , label='μ=1e-6,ε=1e-5')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*2],linestyle=':', label='μ=1e-6,ε=1e-4')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*3], label='μ=1e-5,ε=1e-6')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*4],linestyle='--' , label='μ=1e-5,ε=1e-5')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*5],linestyle=':', label='μ=1e-5,ε=1e-4')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*6], label='μ=1e-4,ε=1e-6')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*7],linestyle='--' , label='μ=1e-4,ε=1e-5')
plt.plot(result_2[:,0], result_2[:,6+result_col_number*8],linestyle=':', label='μ=1e-4,ε=1e-4')
plt.xlabel('s, ($λ=10^{(-4+0.15s)}$)', fontsize=20)
plt.ylabel('Mean Squared Error', fontsize=15)
plt.legend(bbox_to_anchor=(1, 1), fontsize=15)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# 求最小值所在的位置
np.argmin(result_2[:,6+result_col_number*0])
np.argmin(result_2[:,6+result_col_number*1])
np.argmin(result_2[:,6+result_col_number*2])
np.argmin(result_2[:,6+result_col_number*3])
np.argmin(result_2[:,6+result_col_number*4])
np.argmin(result_2[:,6+result_col_number*5])
np.argmin(result_2[:,6+result_col_number*6])
np.argmin(result_2[:,6+result_col_number*7])
np.argmin(result_2[:,6+result_col_number*8])