"""
Created by Janos Szalai-Gindl and Márton Ambrus-Dobai;
"""

import gc
import json
from math import ceil, floor
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import iqr
from QBB_endpoints import Endpoints

import time

GENERATE_STAT = 0
USE_EXISTING_STAT = 1

class Statistics:

	stat_file = "descriptor_statistics.json"
	use_limit = False
	limit = 64
	bit_nums : np.ndarray
	data_buffer : dict
	endpoints : Endpoints

	def __init__(self, stat_file : str = "descriptor_statistics.json", use_limit : bool = True, limit : int = 64) -> None:
		self.stat_file = stat_file
		self.use_limit = use_limit
		self.limit = limit
		if use_limit:
			self.stat_file = str(limit)+"_"+stat_file
		self.row_size = None
		self.data_buffer = {}

	def get_descriptor_statistics(self, other_descriptor_folder : str, search_for_file : int = USE_EXISTING_STAT) -> list:
		if USE_EXISTING_STAT == search_for_file:
			try:
				with open(other_descriptor_folder+'/'+self.stat_file) as f:
					data = json.load(f)
					print("Statistics loaded from:", other_descriptor_folder+'/'+self.stat_file)
					return data
			except IOError:
				return self.create_descriptor_statistics(other_descriptor_folder)
		else:
			return self.create_descriptor_statistics(other_descriptor_folder)

	def create_descriptor_statistics(self, other_descriptor_folder : str) -> list:
		self.endpoints = Endpoints(other_descriptor_folder)
		print('Endpoints loading from', self.endpoints.path, ':', 'succesfull' if self.endpoints.try_load() else 'failed')
		files = [join(other_descriptor_folder, f) for f in listdir(other_descriptor_folder) if isfile(join(other_descriptor_folder, f))]
		descriptor_dims : int
		fname = [file for file in files if '.csv'==file[-4:]][0]
		with open(fname, 'r') as f:
			line = f.readline()
			descriptor_dims = len(line.split(','))
		index_attribs = self._find_endpoints_for_indices(descriptor_dims, files)
		print("---------------------\nBumps")
		print(json.dumps(index_attribs, sort_keys=False, indent=4))
		with open(other_descriptor_folder+'/'+self.stat_file, 'w') as f:
			json.dump(index_attribs, f, sort_keys=False, indent=2)
		return index_attribs

	def _find_endpoints_for_indices(self, dims : int, files : list) -> list:
		index_attribs = []
		print('dims:',dims)

		if self.use_limit:
			_,self.bit_nums = self.get_bit_numbers(dims, files, self.limit)
		for i in range(dims):
			print(i,". index's attribs")
			if self.endpoints.is_loaded() and self.endpoints.has_key(i):
				if self.use_limit:
					index_attribs.append(self.endpoints.get(i, self.bit_nums[i]))
				else:
					index_attribs.append(self.endpoints.get(i))
			else:
				start_f = time.time()
				index_values = self._get_all_index_values_from_files(i, files, dims)
				end_f = time.time()
				start_e = time.time()
				if self.use_limit:
					index_attribs.append(self._find_endpoints_for_index(index_values, self.bit_nums[i]))
				else:
					index_attribs.append(self._find_endpoints_for_index(index_values))
				end_e = time.time()
				print("\tData load time:", end_f-start_f)
				print("\tEndpoint time: ", end_e-start_e)
		return index_attribs

	def _get_all_index_values_from_files(self, idx : int, files : list, dims : int = 100) -> np.ndarray:
		if idx in self.data_buffer.keys():
			return self.data_buffer[idx]
		self.data_buffer.clear()
		gc.collect()
		max_dims = 77
		stop = idx+max_dims if dims > idx+max_dims else dims
		print("index and stop:",idx, stop)
		idxs = np.arange(start=idx, stop=stop)
		data : np.ndarray
		i = 0
		for file in files:
			if file[-4:] == ".csv":
				load_data = np.loadtxt(file, delimiter=",", usecols=idxs.tolist())
				if 0 == i:
					data = load_data
				else:
					data = np.concatenate((data, load_data))
				i += 1
				progress = ceil((i/60)*100)
				sys.stdout.write('\r\tReading files: [{0}] {1}%'.format('#'*(floor(progress/5))+' '*(20-floor(progress/5)), progress))
		sys.stdout.write('\r\tReading files: [{0}] {1}%\n\n'.format('#'*20, 100))
		data = data.T
		n, m = data.shape
		for i in range(n):
			self.data_buffer[idxs[i]] = data[i]
		return data[0]

	def _find_endpoints_for_index(self, values : np.ndarray, bit_num : int = None) -> list:
		level, all_attribs = self._get_level_for_dim(values)

		if bit_num is not None:
			groups = 2**bit_num
			all_attribs = [x for x in all_attribs if groups >= len(x)-1]
		attribs = all_attribs[-1]
		print(attribs)

		return attribs


	def _get_level_for_dim(self, fd, max_bin_numbers = 10000):
		fd_min = np.min(fd)
		fd_max = np.max(fd)
		# Freedman–Diaconis rule:
		bw_by_F_D_r = 2*iqr(fd)/np.cbrt(len(fd))
		if ((fd_max - fd_min)/max_bin_numbers) < bw_by_F_D_r:
			bw = bw_by_F_D_r
		else:
			bw = (fd_max - fd_min)/max_bin_numbers
		level = 0
		min_diff = 1.0
		endpoints = None
		all_endpoints = []
		while(min_diff > 0.0):
			level += 1
			group_number = 2**level
			endpoints = np.array([np.quantile(fd, idx*(1.0 / (group_number))) 
						for idx in range(group_number+1)])

			endpoints = np.rint(endpoints/bw)*bw
			differences_of_endps = [endpoints[idx+1] - endpoints[idx] for idx in range(group_number)]
			min_diff = np.min(differences_of_endps)
			all_endpoints.append(endpoints.tolist())

		result_level = level
		if level > 1:
			result_level = level-1
			all_endpoints = all_endpoints[:-1]

		print('\t',result_level)
		return result_level, all_endpoints

	def get_bit_numbers(self, dim_no, files, capacity = 352, printed = True, gray_coding = True):
		requested_bits = []
		for dim_idx in range(dim_no):
			if self.endpoints.is_loaded() and self.endpoints.has_key(dim_idx):
				level = self.endpoints.get_level(dim_idx)
			else:
				fd = self._get_all_index_values_from_files(dim_idx, files, dim_no)
				fd = fd[~np.isnan(fd)]
				level,endpoints = self._get_level_for_dim(fd)
				self.endpoints.add(dim_idx, level, endpoints)
			if gray_coding:
				requested_bit = level
				requested_bits.append(requested_bit)
			else:
				requested_bit = 2**level - 1
				requested_bits.append(requested_bit)
		requested_bits = np.array(requested_bits)
		all_requested_bits = requested_bits.sum()
		if capacity < all_requested_bits:
			rate = (capacity-dim_no) / (all_requested_bits-dim_no)
			if gray_coding:
				bits_per_dims = np.ones(dim_no)
				bits_per_dims += np.floor((requested_bits - 1)*rate)
			else:
				bits_per_dims = np.zeros(dim_no)
				exponent = np.floor(np.log2((requested_bits/2 - 0.5)*rate + 1)) + 1
				bits_per_dims += (np.power(2, exponent) - 1)
		else:
			bits_per_dims = requested_bits
		bits_per_dims = bits_per_dims.astype(int)
		if printed:
			all_received_bits = bits_per_dims.sum()
			print("all requested bits:", all_requested_bits, "all received bits:", all_received_bits)
			for dim_idx in range(dim_no):
				print("dim_idx:",dim_idx,"requested bit(s):",requested_bits[dim_idx],
					"received bit(s)",bits_per_dims[dim_idx])
		return requested_bits, bits_per_dims