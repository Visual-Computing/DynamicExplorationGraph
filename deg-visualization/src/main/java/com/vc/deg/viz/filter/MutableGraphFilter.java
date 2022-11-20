package com.vc.deg.viz.filter;

import java.util.function.Consumer;
import java.util.function.IntConsumer;

import com.vc.deg.graph.GraphFilter;

public interface MutableGraphFilter extends GraphFilter {
	
//	public MutableGraphFilter add(Consumer<IntConsumer> idProvider);
//	public MutableGraphFilter change(Consumer<IntConsumer> addProvider, Consumer<IntConsumer> removeProvider);

	/**
	 * Make a copy of the filter and removes the ids provided by the id provider.
	 * The id provider gets a removeFunc of type {@link IntConsumer}
	 * 
	 * 
	 * @param idProvider
	 * @return
	 */
	public MutableGraphFilter remove(Consumer<IntConsumer> idProvider);	
}
