package com.vc.deg.ref.graph;

import java.util.function.Consumer;
import java.util.function.IntConsumer;

import com.vc.deg.graph.VertexFilter;

public class VertexFilterFactory implements com.vc.deg.graph.VertexFilterFactory{

	@Override
	public VertexFilter of(int[] validIds, int allElementCount) {
		return new MutableVertexFilter(validIds, allElementCount);
	}
	
	@Override
	public VertexFilter of(Consumer<IntConsumer> validIds, int allElementCount) {
		return new MutableVertexFilter(validIds, allElementCount);
	}

	@Override
	public void and(VertexFilter x1, VertexFilter x2) {
		final MutableVertexFilter m1 = (MutableVertexFilter)x1;
		final MutableVertexFilter m2 = (MutableVertexFilter)x2;
		m1.and(m2);
	}

	@Override
	public void andNot(VertexFilter x1, VertexFilter x2) {
		final MutableVertexFilter m1 = (MutableVertexFilter)x1;
		final MutableVertexFilter m2 = (MutableVertexFilter)x2;
		m1.andNot(m2);
	}

	@Override
	public void add(VertexFilter x1, Consumer<IntConsumer> x2) {
		final MutableVertexFilter m1 = (MutableVertexFilter)x1;
		m1.add(x2);
	}

	@Override
	public void remove(VertexFilter x1, Consumer<IntConsumer> x2) {
		final MutableVertexFilter m1 = (MutableVertexFilter)x1;
		m1.remove(x2);
	}
}
