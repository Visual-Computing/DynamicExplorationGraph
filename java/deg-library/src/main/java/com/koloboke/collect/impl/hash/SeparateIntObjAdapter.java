package com.koloboke.collect.impl.hash;

import com.koloboke.collect.hash.HashConfig;
import com.koloboke.collect.map.IntObjMap;
import com.koloboke.collect.map.hash.HashIntObjMapFactory;
import com.koloboke.collect.map.hash.HashIntObjMaps;

/**
 * Helper class to access the internal data of a IntObjMap
 * 
 * @author Nico Hezel
 */
public class SeparateIntObjAdapter implements SeparateKVIntObjLHash {

	protected HashConfigWrapper configWrapper;
	
	protected int freeValue;
	protected boolean supportRemoved;
	protected int removedValue;
	protected int capacity;
	protected int freeSlots;
	protected int[] keys;
	protected Object[] valueArray;
	protected boolean noRemoved;
	protected int removedSlots;
	protected int modCount;
	protected HashConfig hashConfig;
	protected double currentLoad;
	protected int size;
	protected long sizeAsLong;
	protected boolean isEmpty;
	
	public SeparateIntObjAdapter() {
		// get the default configuration
		HashIntObjMapFactory<?> factory = HashIntObjMaps.getDefaultFactory();
		configWrapper = new HashConfigWrapper(factory.getHashConfig());
	}
	
	public SeparateIntObjAdapter(IntObjMap<?> adapter) {	
		this((SeparateKVIntObjLHash)adapter);
	}

	public SeparateIntObjAdapter(SeparateKVIntObjLHash adapter) {
		freeValue = adapter.freeValue();
		supportRemoved = adapter.supportRemoved();
		removedValue = 0;//adapter.removedValue();
		configWrapper = adapter.configWrapper();
		capacity = adapter.capacity();
		freeSlots = adapter.freeSlots();
		noRemoved = adapter.noRemoved();
		removedSlots = adapter.removedSlots();
		modCount = adapter.modCount();
		hashConfig = adapter.hashConfig();
		currentLoad = adapter.currentLoad();
		size = adapter.size();
		sizeAsLong = adapter.sizeAsLong();
		isEmpty = adapter.isEmpty();
		keys = adapter.keys();
		valueArray = adapter.valueArray();
	}
	
	public <T> IntObjMap<T> immutableIntObjMap() { 
		ImmutableLHashSeparateKVIntObjMapGO<T> res = new ImmutableLHashSeparateKVIntObjMap<>();
		res.move(this);
		return res;
	}
	
	
	public <T> IntObjMap<T> mutableIntObjMap() { 
		return mutableIntObjMap(new MutableLHashSeparateKVIntObjMap<>());
	}
	
	public <T> IntObjMap<T> mutableIntObjMap(MutableLHashSeparateKVIntObjMapGO<T> res) {
		res.move(this);
		return res;
	}
	
	public <T> IntObjMap<T> updatableIntObjMap() { 
		UpdatableLHashSeparateKVIntObjMapGO<T> res = new UpdatableLHashSeparateKVIntObjMap<>();
		res.move(this);
		return res;
	}
	
	@Override
	public int freeValue() {
		return freeValue;
	}

	@Override
	public boolean supportRemoved() {
		return supportRemoved;
	}

	@Override
	public int removedValue() {
		return removedValue;
	}

	@Override
	public HashConfigWrapper configWrapper() {
		return configWrapper;
	}

	@Override
	public int capacity() {
		return capacity;
	}

	@Override
	public int freeSlots() {
		return freeSlots;
	}

	@Override
	public int[] keys() {
		return keys;
	}

	@Override
	public boolean noRemoved() {
		return noRemoved;
	}

	@Override
	public int removedSlots() {
		return removedSlots;
	}

	@Override
	public int modCount() {
		return modCount;
	}

	@Override
	public HashConfig hashConfig() {
		return hashConfig;
	}

	@Override
	public double currentLoad() {
		return currentLoad;
	}

	@Override
	public boolean ensureCapacity(long minSize) {
		return true;
	}

	@Override
	public boolean shrink() {
		return true;
	}

	@Override
	public int size() {
		return size;
	}

	@Override
	public long sizeAsLong() {
		return sizeAsLong;
	}

	@Override
	public void clear() {}

	@Override
	public boolean isEmpty() {
		return isEmpty;
	}

	@Override
	public Object[] valueArray() {
		return valueArray;
	}
	
	public void setValueArray(Object[] valueArray) {
		this.valueArray = valueArray;
	}

	public void setConfigWrapper(HashConfigWrapper configWrapper) {
		this.configWrapper = configWrapper;
	}

	public void setFreeValue(int freeValue) {
		this.freeValue = freeValue;
	}

	public void setSupportRemoved(boolean supportRemoved) {
		this.supportRemoved = supportRemoved;
	}

	public void setRemovedValue(int removedValue) {
		this.removedValue = removedValue;
	}

	public void setCapacity(int capacity) {
		this.capacity = capacity;
	}

	public void setFreeSlots(int freeSlots) {
		this.freeSlots = freeSlots;
	}

	public void setKeys(int[] keys) {
		this.keys = keys;
	}

	public void setNoRemoved(boolean noRemoved) {
		this.noRemoved = noRemoved;
	}

	public void setRemovedSlots(int removedSlots) {
		this.removedSlots = removedSlots;
	}

	public void setModCount(int modCount) {
		this.modCount = modCount;
	}

	public void setHashConfig(HashConfig hashConfig) {
		this.hashConfig = hashConfig;
	}

	public void setCurrentLoad(double currentLoad) {
		this.currentLoad = currentLoad;
	}

	public void setSize(int size) {
		this.size = size;
	}

	public void setSizeAsLong(long sizeAsLong) {
		this.sizeAsLong = sizeAsLong;
	}

	public void setEmpty(boolean isEmpty) {
		this.isEmpty = isEmpty;
	}	
}