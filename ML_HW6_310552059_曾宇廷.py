#!/usr/bin/env python
# coding: utf-8

# In[45]:


import imageio
import numpy as np
import os
from array2gif import write_gif


# In[46]:


def img_formater(img):
    H = img.shape[0]
    W = img.shape[1]
    spatial_data = []
    color_data = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            spatial_data.append([i, j])
            color_data.append(img[i][j])
            
    image_flat = np.zeros((W * H, 3))
    for h in range(H):
        image_flat[h * W:(h + 1) * W] = img[h]
        
    return image_flat, np.array(spatial_data), np.array(color_data, dtype=int)
    
img = imageio.imread('image2.png')
data, spatial_data, color_data = img_formater(img)


# In[47]:


def euclidean(u, v):
    return np.matmul(u**2, np.ones((u.shape[1],v.shape[0])))         -2*np.matmul(u, v.T)         +np.matmul(np.ones((u.shape[0], v.shape[1])), (v.T)**2)

def rbf(u, v, g=10**-4):
    return np.exp(-1*g*euclidean(u, v))
    
gram = rbf(spatial_data, spatial_data) * rbf(color_data, color_data)


# In[48]:


def __get_init(method, data, k):
        if method == 'kmeans++':
            return kmeanspp(data, k)
        elif method == 'default':
            return traditional(data, k)
        else:
            print('ERROR: \'{}\' is not a pre-defined initialize method'.format(method))
            return exit(0)

def kmeanspp(data, k):
    n, d = data.shape
    centers = np.array([data[np.random.randint(n), :d]])
    for i in range(k-1):
        dist = euclidean(data, centers)
        dist = np.min(dist, axis=1)
        next_center = np.argmax(dist, axis=0)
        centers = np.vstack((centers, data[next_center, :]))
    return centers

def traditional(data, k):
    return np.array(data[np.random.choice(data.shape[0], size=k, replace=False), :])


# In[49]:


def kernel_trick(gram, ck):
    c_count = np.sum(ck, axis=0)
    dist = -2*np.matmul(gram, ck)/c_count +         np.matmul(np.ones(ck.shape),(np.matmul(ck.T, np.matmul(gram, ck)))*np.eye(ck.shape[1])) / (c_count**2)
    return dist


# In[10]:


colormap= np.random.choice(range(256),size=(100,3))

def visualize(X,k,H,W):
    '''
    @param X: (10000) belonging classes ndarray
    @param k: #clusters
    @param H: image_H
    @param W: image_W
    @return : (H,W,3) ndarray
    '''
    colors= colormap[:k,:]
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:]=colors[X[h*W+w]]

    return res.astype(np.uint8)


# In[7]:


def save_gif(segments,gif_path):
    for i in range(len(segments)):
        segments[i] = segments[i].transpose(1, 0, 2)
    write_gif(segments, gif_path, fps=2)


# In[8]:


method = 'kmeans++' #try to change to default
k = 3
centers = __get_init(method, gram, k)
print(centers.shape)
dist = euclidean(gram, centers)
print(dist.shape)
ck = np.zeros((gram.shape[0], k))
ck[np.arange(dist.shape[0]), np.argmin(dist, axis=1)] = 1

segments = []
record = []
record.append(ck)
record_iter = 0
max_iter = 50
is_kernel = 0
converge = 5
for i in range(max_iter):
    #E-step
    dist = kernel_trick(gram, ck)

    #M-step
    update_ck = np.zeros(dist.shape)
    update_ck[np.arange(dist.shape[0]),np.argmin(dist, axis=1)] = 1
    delta_ck = np.count_nonzero(np.abs(update_ck - ck))
    
    record.append(update_ck)
    ck = update_ck
    centers = update_centers
    
    # visualize
    C = np.argmax(update_ck, axis = 1)
    segment = visualize(C, k, 100, 100)
    segments.append(segment)
    
    print('iteration {}'.format(i))
    print(delta_ck)
    
    if delta_ck == 0:
        converge -= 1
        if converge == 0:
            record_iter = i+1
            break
#             if 1 == False:  #here also need to change, try not to record 100th CK.
#                 break

    


# In[13]:


img_path='image1.png'
gif_path=os.path.join('GIF','{}_{}Clusters_{}'.format(img_path.split('.')[0],k,'k-means.gif'))
save_gif(segments,gif_path)


# In[50]:


from scipy.linalg import eig
from matplotlib import pyplot as plt
from matplotlib import colors


# In[51]:


def eig(L, D, normalize):
    if normalize:
        sqrt_D = np.sqrt(D)
        neg_sqrt_D = np.linalg.inv(sqrt_D)
        N = np.matmul(np.matmul(neg_sqrt_D, L), sqrt_D)
        eigenvals, eigenvecs = np.linalg.eig(N)
        eigenvecs = np.matmul(neg_sqrt_D, eigenvecs.real)
        return eigenvals, eigenvecs
    else:
        return np.linalg.eig(L)


# In[52]:


def get_sorted_k_eigen(eigenvalues, eigenvectors, k):
    sorted_idx = np.argsort(eigenvalues)
    sorted_eigenvalues = []
    sorted_eigenvectors = []
    for i in range(k):
        vector = eigenvectors[:, sorted_idx[i]]
        sorted_eigenvectors.append(vector[:, None])
        sorted_eigenvalues.append(eigenvalues[sorted_idx[i]])        
    sorted_eigenvalues = np.array(sorted_eigenvalues)
    sorted_eigenvectors = np.concatenate(sorted_eigenvectors, axis=1)
    return sorted_eigenvalues, sorted_eigenvectors


# In[53]:


k_visual = colors.to_rgba_array(['tab:blue','tab:orange','tab:green'                                 ,'tab:red','tab:purple','tab:brown'])
def visualizer(record, save_path, figsize=(100,100,4)):
    print('visualizing.........', end='\r')
    gif = []
    for i in range(len(record)):
        c_id = np.argmax(record[i], axis=1)
        img = np.zeros(figsize, dtype=np.uint8)
        for j in range(c_id.shape[0]):
            m, n = (int(j/100), int(j%100))
            img[m][n] = 255*k_visual[c_id[j]]
        gif.append(img)
    imageio.mimsave(save_path, gif)
    print('visualizing.........[\033[92mcomplete\033[0m]')


# In[58]:


def merge_gifs(gifs, max_fram, id, method):
    gif = []
    for i in range(len(gifs)):
        gif.append(imageio.get_reader('output2/'+gifs[i]+'.gif'))

    new_gif = imageio.get_writer('output2/image'+str(id)+'_'+method+'.gif')
    
    if max_fram + 5 < 100:
        max_fram += 5
    for frame_number in range(max_fram):
        img = []
        for i in range(len(gif)):
            img.append(gif[i].get_next_data())
        new_image = np.hstack(img)
        new_gif.append_data(new_image)
    for i in range(len(gif)):
        gif[i].close()
    new_gif.close()


# In[59]:


W = gram
D = np.sum(gram, axis=1, keepdims=True) * np.eye(gram.shape[0])
L = D - W
eigenvalues, eigenvectors = eig(L, D, 0)


# In[60]:


def spectral_clustering(k_eigenvectors, k, normalize, converge = 5):
#     W = gram
#     D = np.sum(gram, axis=1, keepdims=True) * np.eye(gram.shape[0])
#     L = W - D
#     k_eigenvalues, k_eigenvectors = __get_sorted_k_eigen(L, k, D, normalize)
    
    is_kernel = 0
#     k_eigenvalues, k_eigenvectors = get_sorted_k_eigen(eigenvalues, eigenvectors, k)
    
    method = 'default'
    centers = __get_init(method, k_eigenvectors, k)
    print(centers.shape)
    dist = euclidean(k_eigenvectors, centers)
    print(dist.shape)
    ck = np.zeros((k_eigenvectors.shape[0], k))
    
    record = []
    ck[np.arange(dist.shape[0]), np.random.randint(k, size=dist.shape[0])] = 1
    record.append(ck)
    ck[np.arange(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    record.append(ck)
    record_iter = 0
    for i in range(100):
        #E-step
        if is_kernel:
            print(ck.shape)
            dist = kernel_trick(k_eigenvectors, ck)
        else:
            dist = euclidean(k_eigenvectors, centers)

        #M-step
        update_ck = np.zeros(dist.shape)
        update_ck[np.arange(dist.shape[0]),np.argmin(dist, axis=1)] = 1
        delta_ck = np.count_nonzero(np.abs(update_ck - ck))
        update_centers = np.matmul(update_ck.T, k_eigenvectors)/np.sum(update_ck, axis=0, keepdims=True).T

        record.append(update_ck)

        print('iteration {}'.format(i))
        print(delta_ck)
        
        if delta_ck == 0:
            converge -= 1
            if converge == 0:
                record_iter = i+1            
                if 1 == False:
                    break

        ck = update_ck
        centers = update_centers
    return record, record_iter


# In[27]:


def reorder_by_cluster(c):
    new_order = np.array([])
    for i in range(c.shape[1]):
        new_order = np.append(new_order, np.where(c[:,i]==1)[0])
    new_order = new_order.astype('int32')
    return new_order

def show_vectors_by_clusters(vectors, clusters, fig_path):
    reorder_idx = reorder_by_cluster(clusters)
    num_cluster = np.sum(clusters, axis=0, dtype=int)
    iter_idx = 0
    for k in range(clusters.shape[1]):
        plt.subplot(1, clusters.shape[1], k+1)
        for j in range(num_cluster[k]):
            plt.plot(vectors[reorder_idx[iter_idx], :])
            plt.title('cluster '+str(k+1))
            iter_idx += 1
    plt.savefig(fig_path)
    plt.show(block = False)
    plt.pause(1)
    plt.close()


# In[65]:


img_path = ['dataset/image1.png', 'dataset/image2.png']
gif_path = [['image1spectral clustering(k=3, normal)', 'image1spectral clustering(k=4, normal)', 'image1spectral clustering(k=5, normal)'],     ['image1spectral clustering(k=3, ratio)', 'image1spectral clustering(k=4, ratio)', 'image1spectral clustering(k=5, ratio)'],     ['image2spectral clustering(k=3, normal)', 'image2spectral clustering(k=4, normal)', 'image2spectral clustering(k=5, normal)'],     ['image2spectral clustering(k=3, ratio)', 'image2spectral clustering(k=4, ratio)', 'image2spectral clustering(k=5, ratio)']]

if not os.path.exists('./output2'):
    os.mkdir('./output2')

max_fram = 0
for j in range(3):
#     sc1 = spectral_clustering(similarity, k=3+j, normalize=True, keep_log=True)
#     sc2 = spectral_clustering(similarity, k=3+j, normalize=False, keep_log=True)
    
    k_eigenvalues, k_eigenvectors = get_sorted_k_eigen(eigenvalues, eigenvectors, k=3+j) #新增此行
    
    record_norm, iter_norm = spectral_clustering(k_eigenvectors, k=3+j, normalize=False)
    visualizer(record_norm, 'output2/'+gif_path[3][j]+'.gif')
#     (record_ratio, iter_ratio) = spectral_clustering(gram, k=3+j, normalize=False)
#     visualizer(record_ratio, 'output/'+gif_path[(i*2)+1][j]+'.gif')

    max_fram = max(max_fram, iter_norm)
#     if iter_norm <= iter_ratio:
#         print('faster.....................[\033[94mnormalize\033[0m]')
#     else:
#         print('faster.....................[\033[95munnormalize\033[0m]')

merge_gifs(gif_path[3], max_fram, 2, 'unnormalize')
# merge_gifs(gif_path[(i*2)+1], max_fram, 1, 'unnormalize')


# In[ ]:




