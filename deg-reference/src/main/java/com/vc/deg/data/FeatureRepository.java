package com.vc.deg.data;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Map;

import com.vc.deg.data.DataRepository;
import com.vc.deg.data.FeatureSpace;

public class FeatureRepository<T> implements DataRepository<T> {

	protected Map<Integer, T> map;
	protected FeatureSpace<T> space;
	
	public FeatureRepository(FeatureSpace<T> space) {
		this.map = new HashMap<>();
		this.space = space;
	}
	
	public FeatureRepository(FeatureSpace<T> space, int expectedSize) {
		this.map = new HashMap<>(expectedSize);
		this.space = space;
	}

	@Override
	public FeatureSpace<T> getFeatureSpace() {
		return space;
	}

	@Override
	public T get(int id) {
		return map.get(id);
	}

	@Override
	public void add(int id, T obj) {
		map.put(id, obj);
	}

	public void writeObject(ObjectOutputStream out) throws IOException {

	}

	public void readObject(ObjectInputStream in) throws IOException {

	}
}
