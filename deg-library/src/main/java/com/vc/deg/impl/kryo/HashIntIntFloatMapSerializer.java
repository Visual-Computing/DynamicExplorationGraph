package com.vc.deg.impl.kryo;


import java.io.BufferedInputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.koloboke.collect.impl.hash.MutableLHashSeparateKVIntObjMapGO;
import com.koloboke.collect.impl.hash.SeparateIntObjAdapter;
import com.koloboke.collect.map.IntFloatMap;
import com.koloboke.collect.map.IntObjMap;
import com.koloboke.collect.map.hash.HashIntFloatMap;

public class HashIntIntFloatMapSerializer extends Serializer<IntObjMap<IntFloatMap>> {

	public static final byte NULL = 0;
	public static final byte NOT_NULL = 1;
	
	@Override
	public void write(Kryo kryo, Output output, IntObjMap<IntFloatMap> object) {
		store(output, object);
	}

	@Override
	public IntObjMap<IntFloatMap> read(Kryo kryo, Input input, Class<? extends IntObjMap<IntFloatMap>> type) {
		return loadMutable(input);
	}
	
	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------
	
	
	public static void store(Output output, IntObjMap<IntFloatMap> obj) {

		SeparateIntObjAdapter adapter = new SeparateIntObjAdapter(obj);
		output.writeInt(adapter.keys().length);
		output.writeInts(adapter.keys(), 0, adapter.keys().length);
		saveIntFloatMapArray(output, cast(adapter.valueArray()));
		output.writeInt(adapter.freeValue());
		output.writeInt(adapter.size());
		output.writeInt(adapter.capacity());
		output.writeInt(adapter.modCount());
	}

	private static HashIntFloatMap[] cast(Object[] values) {
		HashIntFloatMap[] result = new HashIntFloatMap[values.length];
		for (int i = 0; i < values.length; i++)
			result[i] = (HashIntFloatMap)values[i];
		return result;
	}
	
	private static void saveIntFloatMapArray(Output output,  HashIntFloatMap[] arr) {
		
		output.writeInt(arr.length, true);
		for (HashIntFloatMap map : arr) {
			
			// überspringe null einträge
			output.writeByte((map == null) ? NULL : NOT_NULL);
			if(map == null)
				continue;
		
			HashIntFloatMapSerializer.store(output, map);
		}
	}
	
	
	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------

	public static IntObjMap<IntFloatMap> loadMutable(Input input) {		
		return load(input).mutableIntObjMap();
	}
	
	public static IntObjMap<IntFloatMap> loadMutable(Input input, IntObjMap<IntFloatMap> map) {
		final MutableLHashSeparateKVIntObjMapGO<IntFloatMap> res = (MutableLHashSeparateKVIntObjMapGO<IntFloatMap>) map;
		return load(input).mutableIntObjMap(res);
	}
	
	private static SeparateIntObjAdapter load(Input input) {
		SeparateIntObjAdapter adapter = new SeparateIntObjAdapter();

		// read the data from file
		int tableSize = input.readInt();
		adapter.setKeys(input.readInts(tableSize));
		adapter.setValueArray(loadIntFloatMapArray(input));
		adapter.setFreeValue(input.readInt());
		adapter.setSize(input.readInt());
		adapter.setCapacity(input.readInt());
		adapter.setModCount(input.readInt());
		
		return adapter;
	}
	
	private static IntFloatMap[] loadIntFloatMapArray(Input input) {
		
		int length = input.readInt(true);
		if (length == 0) return null;

		IntFloatMap[] result = new IntFloatMap[length];
		for (int i = 0; i < length; i++) {
			
			// wenn das object null ist
			if(input.readByte() == NULL)
				continue;
			
			result[i] = HashIntFloatMapSerializer.load(input);
		}
		
		return result;
	}
}
