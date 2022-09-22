import os
import shutil
from os import listdir
from os.path import isfile, join

class DataStore:
  def __init__ (self, data_path, name=None):
    self.name = name
    self.data_path  = data_path

  def info(self):
    '''This method returns information about the datastore
    '''
    if self.name is not None:
      print('Name: %s ' % self.name)
      print('Data path: %s' %self.data_path)

  def list_of_animals(self):
    ''' This method returns a list with the names of datasets(animals).
    '''
    directory_contents = os.listdir(self.data_path)

    datasets = [item for item in directory_contents if os.path.isdir(self.data_path + '/' + item)]
    return datasets

  def list_of_animals_dir(self):
    ''' This method return a list of directories in datastore
    '''
    list_of_subdir = [x[0] for x in os.walk(self.data_path)]
    list_of_subdir.pop(0)

    return list_of_subdir

  def isin(self, dataset_name):
    ''' This method returns True if the given dataset is 
    into datastore, False otherwise.
    
    Inputs:
    dataset_name (str): a valid dataset name

    Output:
    Boolean: True or False
    '''
    sub_dir = [x[0] for x in os.walk(self.data_path)]
    input_dir = self.data_path +'/'+ str(dataset_name)

    return input_dir in sub_dir
  
  def copydata(self, dst):
    ''' This method copy a file named src to another path named dst

    Inputs:
    sdt (str): The directory path that you want copy the data, the directory must be new.

    Output:
    None

    '''
    src = self.data_path

    shutil.copytree(src, dst)
  
  def copyanimal(self, animal, dst):
    ''' This method copy a animal data to another path named dst
    '''
    if self.isin(animal):
      dir = os.path.join(self.data_path, animal)
      shutil.copytree(dir, dst)
    else:
      raise Exception('Invalid animal')
    
  def nanimals(self):
    ''' This method return number of animals in the datastore
    '''

    return len(self.list_of_animals())
    

  def size(self):
    ''' 
      This method return the size of every single datasets
    '''

    #FIXME dar um jeito nesse for
    labels = self.list_of_animals_dir()
    size = 0
    for item in labels:
      for path, dirs, files in os.walk(item):
          for f in files:
              fp = os.path.join(path, f)
              size += os.path.getsize(fp)
    return size

  def animals_shanks(self):
    '''This method returns a dictionary, that the animal name is the key
    and the shanks names is the value.
    '''
    animals_dir = self.list_of_animals_dir()
    animals = self.list_of_animals()
    shanks = {}

    #FIXME dar um jeito nesse for
    for n in range(len(animals_dir)):
      files = [f for f in listdir(animals_dir[n]) if isfile(join(animals_dir[n], f))]
      shanks[animals[n]] = files

    return shanks

  def animals_shanks_dir(self):
    '''This method returns a dictionary, that the animal name is the key
    and the shanks directories is the value.
    '''
    animals_dir = self.list_of_animals_dir()
    animals = self.list_of_animals()
    shanks = {}

    #FIXME dar um jeito nesse for
    for m in range(len(animals_dir)):
      shanks_dirs = []
      shanks_labels = [f for f in listdir(animals_dir[m]) if isfile(join(animals_dir[m], f))]
      for n in range(len(shanks_labels)):
        shank_datapath = self.data_path + '/' + animals[m] + '/' + shanks_labels[n]
        shanks_dirs.append(shank_datapath)
      shanks[animals[m]] = shanks_dirs

    return shanks

  def datastore_summary(self):
    '''this method return a DataStore summary
    '''
    animal = []
    shank = []
    n_sua = []
    n_mua = []
    duration = []
    shanks = self.animals_shanks_dir()
    for key in shanks.keys():
      for value in shanks[key]:
        if value[-4:] == 'kwik':
          LP = LocalPopulation(value, name = 'Temp')
          animal_k = key
          shank_k = value[-8:-5]
          n_sua_k = len(LP.get_sua_clu())
          n_mua_k = len(LP.get_mua_clu())
          duration_k = round(LP.t[-1])
          animal.append(animal_k)
          shank.append(shank_k)
          n_sua.append(n_sua_k)
          n_mua.append(n_mua_k)
          duration.append(duration_k)

    dictionary = {'animal':animal, 'shank':shank, 'N SUA':n_sua, 'N MUA':n_mua, 'duração(s)':duration}

    df = pd.DataFrame.from_dict(dictionary)
    return df




from numpy.ma.core import nonzero
from klusta.kwik import KwikModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LocalPopulation:
  def __init__ (self, kpath, name):
    self.name = name
    self.kpath = kpath
    self.kmodel=KwikModel(kpath)
    self.clu=self.kmodel.spike_clusters
    self.t=self.kmodel.spike_times

  def info(self):
    if self.name is not None:
      print ('Name: %s ' % self.name)
      print ('Data path: %s' %self.kpath)

  def get_cluster_groups(self):
    return self.kmodel.cluster_groups


  def get_non_noise_clusters(self):
    ''' Returns a list of non noise clusters
    INPUT
      * LocalPopulation: instance of LocalPopulation class

    OUTPUT
      * list: List of non noise clusters ids
    '''
    G=self.kmodel.cluster_groups

    non_noise = [value for value in G.keys() if G[value]!="noise"]
    return (non_noise)

  def get_pop_activity (self):
    ''' Returns a pair of lists of events taken as spike times. Thus, it will not take 
    events considered as noise.
    INPUT
      * LocalPopulation: instance of LocalPopulation class

    OUTPUT
      * tuple(time, cluster): Lists of time and clu for events taken as spike times
    '''

    non_noise_clusters = self.get_non_noise_clusters()
    mask = np.isin(self.clu, non_noise_clusters)
    clu=self.clu[mask]
    t=self.t[mask]
    
    return (t,clu)

  def get_sua_clu(self):
      ''' Returns a lists of labels corresponding to SUAs within KWIK file.
      INPUT
        * LocalPopulation: instance of LocalPopulation class

      OUTPUT
        * list: List of labels corresponding to SUAs
      '''
      G=self.kmodel.cluster_groups
      
      sua = [value for value in G.keys() if G[value]!='noise' and G[value]!='mua']
      return (sua)

  def get_mua_clu(self):
    ''' Returns a lists of labels corresponding to MUAs within KWIK file.
    INPUT
      * LocalPopulation: instance of LocalPopulation class

    OUTPUT
      * list: List of labels corresponding to SUAs
    '''
    G=self.kmodel.cluster_groups

    mua = [value for value in G.keys() if G[value]=='mua']
    return (mua)

  def get_clu (self, clu_group):
      ''' Returns a lists of labels corresponding to unitary activity (SUA or MUA) 
          within KWIK file.
      INPUT
        * LocalPopulation: instance of LocalPopulation class
        * clu_group (str): Name of the aimed cluster group(SUA or MUA)

      OUTPUT
        * list: Lists of labels corresponding to the aimed unitary activity
      '''

      if not(clu_group in ['SUA', 'MUA']):
        raise Exception('Invalid cluster group, please choose SUA or MUA group.') 
      if (clu_group=='SUA'):
        return (self.get_sua_clu())
      
      return (self.get_mua_clu())
  
  def clu_label(self, clu):
    ''' Returns the type of an aimed cluster.
    INPUT
    * clu: Cluster name

    OUTPUT
    * str: Cluster type(GOOD, MUA or NOISE)
    '''
    G=self.kmodel.cluster_groups

    return G[clu]

  def get_spikes_clu(self, clu_label):
    '''Returns the spike times of a given cluster label
    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * clu_label (int): aimed cluster label.

    OUTPUT
      * tuple (t, clu): t = spike times, in seconds, of a given cluster label.
      clu = corresponding cluster.
    '''
    mask=np.isin(self.clu, clu_label)
    clu=self.clu[mask]
    t=self.t[mask]

    return (t, clu)

  def get_spikes_clus(self, clus_labels):
    '''Returns two lists containing the spikes times and
    the corresponding cluster of the spike event, for a group of clusters belongings a given list
    INPUT
    * clus_labels (list): list of aimed clusters

    OUTPUT
    * tuple(time, clusters): tuple containing times lists and corresponding clusters list of spike events. 
    '''
    mask=np.isin(self.clu, clus_labels)
    clu=self.clu[mask]
    t=self.t[mask]

    return (t, clu)

  def n_SUA(self):
    '''
    This method return the number of SUA neurons
    INPUT
      * LocalPopulation: instance of LocalPopulation class

    OUTPUT
      * int: number of SUA clusters.
    '''

    return len(self.get_sua_clu())

  def n_MUA(self):
    '''
    This method return the number of MUA neurons
    INPUT
      * LocalPopulation: instance of LocalPopulation class.

    OUTPUT
      * int: number of MUA clusters.
    '''

    return len(self.get_mua_clu())

  def n_clu(self, clu_group):
    '''
    This method return the number of units of aimed unit activity type(SUA or MUA).

    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * clu_gorup (str): unit activity type(SUA or MUA)
    OUTPUT
      * int: number of units in the aimed unit activity type.
    
    '''
    if (clu_group not in ['SUA', 'MUA']):
      raise Exception('Invalid cluster gropu, please choose SUA or MUA group.') 
    if (clu_group=='SUA'):
      return (self.n_SUA())
    if (clu_group=='MUA'):
      return (self.n_MUA())

  def IFR(self, bin_size=0.05, clus = None, a=0, b=None):
    '''
    This method return a instant firing rate(IFR) pandas series that Index is
    the bins(default size = 0.05s) of the record duration and the value is the 
    corresponding IFR on that crop of time.

    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * bin_size (float): size of the temporal bin
      * clus (list): list of clusters labels
      * a (float): initial time
      * b (float): final time

    OUTPUT
      * pd.Series: Pandas series
    
    '''
    if clus is None:
      (t,clu) = self.get_pop_activity()
    else:
      (t,clu) = self.get_spikes_clus(clus_labels=clus)

    if b is None:
        b = self.t[-1]

    n_neurons=len(np.unique(clu))
    bins = np.arange(a, b, step = bin_size)
    count, bins = np.histogram(t, bins = bins)
    ifr = pd.Series(count/(bin_size*n_neurons), index = bins[0:-1])

    return ifr

      



  def CV (self, bin_size = 0.05, clus=None, a=0, b=None, window_size=10):
    '''
    This method return a Coefficient of Variation(CV) pandas series that Index is
    the window size(default size = 10s) of the record duration and the value is the 
    corresponding CV on that crop of time.

    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * bin_size (float): size of the temporal bin for IFR calculating
      * a (float): initial time
      * b (float): final time
      * window_size (float): size of the temporal window for CV calculating

    OUTPUT
      * pd.Series: Pandas series
    
    '''
    
    if clus is None:
      (t,clu) = self.get_pop_activity()
    else:
      (t,clu) = self.get_spikes_clus(clus_labels=clus)

    if b is None:
      b = self.t[-1]

    window = np.arange(a, b, step = window_size)
    cv = pd.Series(index = window, dtype='float64')
    ifr = self.IFR(bin_size=bin_size, clus = clus, a=a, b=b)

    for t in window:
      samples = ifr.loc[t:t+window_size]
      mu = np.mean(samples)
      std = np.std(samples)
      if mu != 0:
        cv_t = std/mu
        cv.loc[t] = cv_t

    return cv

  def get_crop(self, b, a=0, t = None, clu = None):
    '''
    This method return a crop of the populational activity record, returning the
    spike times and corresponding cluster for a desired interval between a and b.

    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * a (float): initial time
      * b (float): final time

    OUTPUT
      * tuple(time, clusters): tuple containing times list and corresponding clusters list of spike events.
    
    '''
    if ((t is None) or (clu is None)):
      (t,clu) = self.get_pop_activity()
    
    return ((t[(t>=a) & (t<=b)]),(clu[(t>=a) & (t<=b)]))

  def FR(self, a=0, b=None):
    '''
    This method return a Firing Rate(FR) pandas series that Index is
    the population activity clusters and the value is the 
    corresponding FR on that cluster, on an interval between a and b.

    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * a (float): initial time
      * b (float): final time

    OUTPUT
      * pd.Series: Pandas series
    
    '''

    (t,clu) = self.get_pop_activity()

    if (b is None):
      b = self.t[-1]

    (t_crop,clu_crop) = self.get_crop(a=a,b=b)
    duration = b - a
    clus = np.unique(clu_crop)
    FR = pd.Series(index=clus, dtype=float)
    for k in clus:
      t_k = t_crop[clu_crop==k]
      FR.loc[k] = len(t_k)/duration
    return (FR)

  def FR_max(self, a=0, b=None):
    '''
    This method return a tuple (cluster, value) that cluster is the cluster
    at occurred the maximum Firing Rate(FR) between an interval a and b, and the value
    is that FR.

    INPUT
      * LocalPopulation: instance of LocalPopulation class
      * a (float): initial time
      * b (float): final time

    OUTPUT
      * Tuple(clu, FR): clu = cluster that occurred the max FR, FR = max FR value
    
    '''

    return ((self.FR(a=a, b=b).idxmax()),(self.FR(a=a,b=b).max()))

  def improve_axis (self, ax, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45):
    '''This method improves a axis.
    '''
    
    if title is not None:
        ax.set_title(title, fontsize=18)
    if (xlabel is None):
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        
    if (ylabel is None):
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    else:
        ax.set_ylabel(ylabel, fontsize=18)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)

    for label in (ax.get_xticklabels()):
        label.set_rotation(xrotation)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    return (ax)

  def plot_IFR (self, ifr, ax=None, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45, ymax=None):
    '''
    This metod return a plot of a IFR pandas series

    INPUT
    * ifr (pd.Series): IFR temporal series.

    OUTPUT
    plot of the given IFR temporal series
    '''
    if (ax is None):
      ax = plt.subplot(1,1,1)
    if not(ymax is None):
      ax.set_ylim([0, ymax])
      
    ifr.plot(ax=ax)
    self.improve_axis(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize ,fontname=fontname,xrotation=xrotation) 
      


  def plot_CV (self, cv, ax=None, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45, ymax=None):
    '''
    This metod return a plot of a CV pandas series

    INPUT
    * cv (pd.Series): CV temporal series.

    OUTPUT
    plot of the given CV temporal series
    '''
    if (ax is None):
      ax = plt.subplot(1,1,1)
    if not(ymax is None):
      ax.set_ylim([0, ymax])
      
    cv.plot(ax=ax)
    self.improve_axis(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize ,fontname=fontname,xrotation=xrotation) 

  def population_summary(self, clu_label = None):

    if clu_label == None:
      clus = self.get_non_noise_clusters()
    else:
      clus = self.get_clu(clu_label)
    
    clu_label = []
    n_spikes = []
    duration = []
    FR = []
    duration = []
    contamination = []

    for clu in clus:
      t, trash = self.get_spikes_clu(clu)

      contamination_k = self.get_contamination_rate(clu)
      clu_label_k = self.clu_label(clu)
      n_spikes_k = len(t)
      duration_k = round(t[-1])
      FR_k = n_spikes_k/duration_k
      clu_label.append(clu_label_k)
      n_spikes.append(n_spikes_k)
      duration.append(duration_k)
      FR.append(FR_k)
      contamination.append(contamination_k)

    dictionary = {'Cluster':clus, 'Tipo de cluster':clu_label, 'número de spikes':n_spikes, 'duração(s)':duration, 'FR(spk/s)':FR, 'Taxa de Contaminação(%)':contamination}

    df = pd.DataFrame.from_dict(dictionary)
    df = df.set_index('Cluster')
    return df

  def get_isi(self, clu_label):
    t, trash = self.get_spikes_clu(clu_label)
    isi = np.diff(t)
    return isi
  
  def get_contamination_rate(self, clu_label, limit = 0.001):
    isi = self.get_isi(clu_label)
    count = len(isi[isi<limit])
    contamination = round((count/len(isi)),2)
    return contamination


class Dataset:
  def __init__ (self, dataset_name):
    self.name = dataset_name
    self.local_populations  = {}

  def info(self):
    if self.name is not None:
      print ('Name: %s ' % self.name)

  def add_population(self, LP):
    self.local_populations[LP.name]=LP

  def remove_population(self, LP):
    self.local_populations.pop(LP.name, None)

  def local_population_list (self):

    return self.local_populations.keys()
 
  def nSUA (self):
    nSUA = {}
    for k in self.local_populations.keys():
      nSUA[k] = self.local_populations[k].n_SUA()
    
    return (nSUA)

  def nMUA (self):
    # TODO see if fix works
    
    # nMUA = {}
    # for k in self.local_populations.keys():
    #   nMUA[k] = self.local_populations[k].n_MUA()
    nMUA = {key:value.n_MUA() for (key,value) in self.local_populations.items()}

    return (nMUA)

  def dataset_summary(self):
    d = {'SUA': self.nSUA().values(), 'MUA': self.nMUA().values()}
    df = pd.DataFrame(index = self.nSUA().keys(), data = d)
    df.loc['Total']= df.sum(numeric_only=True, axis=0)
    df.loc[:,'Total'] = df.sum(numeric_only=True, axis=1)
    df = df.sort_index()

    return (df)
  
  def IFR(self, bin_size=0.05, a=0, b=None):
    ifr = {}

    #FIXME dar um jeito nesse for
    for k in self.local_populations.keys():
      if b is None:
        (t,trash) = self.local_populations[k].get_pop_activity()
        b=t[-1]
      ifr[k] = self.local_populations[k].IFR(bin_size=bin_size, a=a, b=b)
    return (ifr)

  def plot_IFR(self, ifr_dict, ax=None, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45, ymax=None):
    if ax is None:
      ax = plt.subplot(1,1,1)

    if not(ymax is None):
      ax.set_ylim([0, ymax])

    for k in ifr_dict.keys():
      ifr = ifr_dict[k]  
      self.local_populations[k].plot_IFR(ifr=ifr, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize ,fontname=fontname,xrotation=xrotation, ymax=ymax)
    ax.legend(labels = self.local_populations.keys())

  def CV(self, bin_size = 0.05, a=0, b=None, window_size=10):

    #FIXME dar um jeito nesse for
    cv = {}
    for k in self.local_populations.keys():
      if b is None:
        (t,trash) = self.local_populations[k].get_pop_activity()
        b=t[-1]
      cv[k] = self.local_populations[k].CV(bin_size=bin_size, a=a, b=b, window_size=window_size)
    return (cv)

  def plot_CV(self, cv_dict, ax=None, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45, ymax=None):
    if ax is None:
      ax = plt.subplot(1,1,1)

    if not(ymax is None):
      ax.set_ylim([0, ymax])

    for k in cv_dict.keys():
      cv = cv_dict[k]  
      self.local_populations[k].plot_CV(cv=cv, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize ,fontname=fontname,xrotation=xrotation, ymax=ymax)
    ax.legend(labels = self.local_populations.keys())