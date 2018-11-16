

import numpy as np

ub=2 #upper bound
n=10 #number of buckets

DSA=np.random.rand(10)
#DSA=DSA*2
bucket=np.array([[-1,-1,0]])

for i in range(1,n+1):
    temp=np.array([[ub*(i-1)/n,ub*i/n,0]])
    bucket=np.concatenate((bucket,temp),axis=0)

bucket=np.delete(bucket, 0, 0)
print('bucket shape',bucket.shape)
#print(bucket)


for i in range(bucket.shape[0]):
    for j in range(len(DSA)):
        if(DSA[j]>bucket[i][0] and DSA[j]<bucket[i][1]):
            print('DSA[',j,']',DSA[j],'bucket[',bucket[i][0],',',bucket[i][1],']')
            bucket[i][2]=1

count_bucket=0
for i in range(bucket.shape[0]): 
    if(bucket[i][2]==1):
        count_bucket=count_bucket+1
                   
print('count_bucket',count_bucket)
DSC=count_bucket/n
print('DSC',DSC)