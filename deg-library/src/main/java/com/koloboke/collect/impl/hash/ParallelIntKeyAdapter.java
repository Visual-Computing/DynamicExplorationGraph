package com.koloboke.collect.impl.hash;

import com.koloboke.collect.hash.HashConfig;
import com.koloboke.collect.map.IntFloatMap;
import com.koloboke.collect.map.IntIntMap;
import com.koloboke.collect.map.hash.HashIntFloatMapFactory;
import com.koloboke.collect.map.hash.HashIntFloatMaps;
import com.koloboke.collect.set.IntSet;


public class ParallelIntKeyAdapter implements ParallelKVIntLHash {

	protected HashConfigWrapper configWrapper;
	
	protected int freeValue;
	protected boolean supportRemoved;
	protected int removedValue;
	protected int capacity;
	protected int freeSlots;
	protected long[] table;
	protected boolean noRemoved;
	protected int removedSlots;
	protected int modCount;
	protected HashConfig hashConfig;
	protected double currentLoad;
	protected int size;
	protected long sizeAsLong;
	protected boolean isEmpty;
	
	// get the default configuration
	public ParallelIntKeyAdapter() {
		HashIntFloatMapFactory factory = HashIntFloatMaps.getDefaultFactory();
		configWrapper = new HashConfigWrapper(factory.getHashConfig());
	}
	
	public ParallelIntKeyAdapter(IntIntMap adapter) {	
		this((ParallelKVIntHash)adapter);
	}
	
	public ParallelIntKeyAdapter(IntFloatMap adapter) {	
		this((ParallelKVIntHash)adapter);
	}
	
	public ParallelIntKeyAdapter(IntSet adapter) {	
		this((ParallelKVIntHash) adapter);
	}
	
	public ParallelIntKeyAdapter(ParallelKVIntHash adapter) {		
		
		// com.koloboke.collect.impl.hash.ParallelKVIntHash
		table = adapter.table();
		
		// com.koloboke.collect.impl.hash.IntHash
		freeValue = adapter.freeValue();
		supportRemoved = adapter.supportRemoved();
		removedValue = supportRemoved ? adapter.removedValue() : 0;
		
		// com.koloboke.collect.impl.hash.Hash
		configWrapper = adapter.configWrapper();
		capacity = adapter.capacity();
		freeSlots = adapter.freeSlots();
		noRemoved = adapter.noRemoved();
		removedSlots = adapter.removedSlots();
		modCount = adapter.modCount();		
		
		// com.koloboke.collect.hash.HashContainer
		hashConfig = adapter.hashConfig();
		currentLoad = adapter.currentLoad();
		
		// com.koloboke.collect.Container
		size = adapter.size();
		sizeAsLong = adapter.sizeAsLong();
		isEmpty = adapter.isEmpty();
		
		// TODO store type
//		ImmutableParallelKVIntQHashSO
//		MutableParallelKVIntQHashSO	
//		UpdatableParallelKVIntQHashSO	
//		ImmutableParallelKVIntLHashSO	
//		MutableParallelKVIntLHashSO	
//		UpdatableParallelKVIntLHashSO	
		// und 
//		IntIntMap
//		IntFloatMap
	}
	
	public IntFloatMap immutableIntFloatMap() { 
		ImmutableLHashParallelKVIntFloatMapGO res = new ImmutableLHashParallelKVIntFloatMap();
		move(res);
		return res;
	}
	
	public IntIntMap immutableIntIntMap() { 
		ImmutableLHashParallelKVIntIntMapGO res = new ImmutableLHashParallelKVIntIntMap();
		move(res);
		return res;
	}
	
	public void move(MutableLHashParallelKVIntKeyMap hash) {
		hash.move(this);
	}
	
	public void move(ImmutableParallelKVIntLHashSO hash) {
		hash.move(this);
	}
	
	public void move(UpdatableParallelKVIntLHashSO hash) {
		hash.move(this);
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
	public long[] table() {
		return table;
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

	public void setTable(long[] table) {
		this.table = table;
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
	
	public static boolean supported(IntSet adapter) {
		return (adapter instanceof ParallelKVIntHash);
	}	
}
