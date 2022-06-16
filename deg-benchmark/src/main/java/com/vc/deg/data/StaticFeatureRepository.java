package com.vc.deg.data;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.vc.deg.data.DataComparator;
import com.vc.deg.data.DataRepository;
import com.vc.deg.data.FeatureSpace;

public class StaticFeatureRepository<T> implements DataRepository<T> {

	protected T[] data;
	protected final FeatureSpace<T> space;
	protected final DataComparator comparator;
	
	public StaticFeatureRepository(FeatureSpace<T> space, T[] data) {
		this.data = data;
		this.space = space;
		this.comparator = this::compare;
	}
	
	public double compare(int id1, int id2) {
		T obj1 = data[id1];
		T obj2 = data[id2];
		return space.computeDistance(obj1, obj2);
	}

	@Override
	public DataComparator getComparator() {
		return comparator;
	}

	@Override
	public void add(int id, T obj) {
		// do nothing		
	}
	
	@Override
	public void write(Kryo kryo, Output output) {
		// do nothing	
	}
	
	@Override
	public void read(Kryo kryo, Input input) {
		// do nothing	
	}

	@Override
	public FeatureSpace<T> getFeatureSpace() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public T get(int id) {
		// TODO Auto-generated method stub
		return null;
	}
}
