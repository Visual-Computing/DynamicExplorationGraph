package com.vc.deg.impl.kryo;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.koloboke.collect.impl.hash.MutableLHashParallelKVIntFloatMapGO;
import com.koloboke.collect.impl.hash.ParallelIntKeyAdapter;
import com.koloboke.collect.map.IntFloatMap;

public class HashIntFloatMapSerializer extends Serializer<IntFloatMap> {

	@Override
	public void write(Kryo kryo, Output output, IntFloatMap obj) {
		store(output, obj);
	}
	
	@Override
	public IntFloatMap read(Kryo kryo, Input input, Class<? extends IntFloatMap> type) {
		return load(input);
	}
	
	public static void store(Output output, IntFloatMap obj) {
		
		ParallelIntKeyAdapter adapter = new ParallelIntKeyAdapter(obj);
		output.writeInt(adapter.table().length);
		output.writeLongs(adapter.table(), 0, adapter.table().length);
		output.writeInt(adapter.freeValue());
		output.writeInt(adapter.size());
		output.writeInt(adapter.capacity());
		output.writeInt(adapter.modCount());
	}

	public static IntFloatMap load(Input input) {

		// create an adapter to fill everything in the final object
		ParallelIntKeyAdapter adapter = new ParallelIntKeyAdapter();
				
		// read the data from file
		int tableSize = input.readInt();
		adapter.setTable(input.readLongs(tableSize));
		adapter.setFreeValue(input.readInt());
		adapter.setSize(input.readInt());
		adapter.setCapacity(input.readInt());
		adapter.setModCount(input.readInt());
		
		// create empty object and copy all data over
		MutableLHashParallelKVIntFloatMapGO result = new MutableLHashParallelKVIntFloatMapGO();
		adapter.move(result);
		
		return result;
	}
}
