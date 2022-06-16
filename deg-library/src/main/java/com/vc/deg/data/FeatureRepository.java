package com.vc.deg.data;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.koloboke.collect.map.IntObjMap;
import com.koloboke.collect.map.hash.HashIntObjMaps;
import com.vc.deg.data.DataRepository;
import com.vc.deg.data.FeatureSpace;

public class FeatureRepository<T> implements DataRepository<T> {

	protected IntObjMap<T> map;
	protected FeatureSpace<T> space;
	protected DataComparator comparator;
	
	public FeatureRepository(FeatureSpace<T> space) {
		this.map = HashIntObjMaps.newMutableMap();
		this.space = space;
		this.comparator = this::compare;
	}
	
	public FeatureRepository(FeatureSpace<T> space, int expectedSize) {
		this.map = HashIntObjMaps.newMutableMap(expectedSize);
		this.space = space;
		this.comparator = this::compare;
	}
	
	public double compare(int id1, int id2) {
		T obj1 = map.get(id1);
		T obj2 = map.get(id2);
		return space.computeDistance(obj1, obj2);
	}
	
	@Override
	public FeatureSpace<T> getFeatureSpace() {
		return space;
	}

	@Override
	public DataComparator getComparator() {
		return comparator;
	}

	@Override
	public void add(int id, T obj) {
		map.put(id, obj);
	}

	@Override
	public void write(Kryo kryo, Output output) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void read(Kryo kryo, Input input) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public T get(int id) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
}
