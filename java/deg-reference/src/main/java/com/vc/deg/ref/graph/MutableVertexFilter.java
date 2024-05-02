package com.vc.deg.ref.graph;

import java.util.function.Consumer;
import java.util.function.IntConsumer;

import org.roaringbitmap.RoaringBitmap;
import org.roaringbitmap.RoaringBitmapAdapter;

import com.vc.deg.graph.VertexFilter;

/**
 * A prepared filter provides a list of valid ids.
 * 
 * @author Nico Hezel
 */
public class MutableVertexFilter extends RoaringBitmap implements VertexFilter {
	
	protected int validIdCount;
	protected int allElementCount;
	
	public MutableVertexFilter() {
		validIdCount = 0;
		allElementCount = 0;
	}
	
	public MutableVertexFilter(Consumer<IntConsumer> validIds, int allElementCount) {
		add(validIds);
		this.allElementCount = allElementCount;
	}
	
	public MutableVertexFilter(int[] validIds, int allElementCount) {
		super();
		RoaringBitmapAdapter.move(RoaringBitmap.bitmapOfUnordered(validIds), this);
		this.allElementCount = allElementCount;
		this.validIdCount = validIds.length;
	}
	
	public MutableVertexFilter(RoaringBitmap validIds, int allElementCount) {
		super();
		RoaringBitmapAdapter.move(validIds, this);		
		this.allElementCount = allElementCount;
		this.validIdCount = validIds.getCardinality();
	}
	
	public void and(MutableVertexFilter x2) {
		super.and(x2);
		this.validIdCount = getCardinality();
	}	
	
	
	public void andNot(MutableVertexFilter x2) {
		super.andNot(x2);
		this.validIdCount = getCardinality();
	}
	
	public void add(Consumer<IntConsumer> v) {
		v.accept(label -> add(label));
	}
	
	public void remove(Consumer<IntConsumer> v) {
		v.accept(label -> remove(label));
	}
	
	@Override
	public void add(int x) {
		validIdCount++;
		super.add(x);
	}
	
	@Override
	public void remove(int x) {
		validIdCount--;
		super.remove(x);
	}
	
	@Override
	public void forEachValidId(IntConsumer action) {
		forEach((int i) -> action.accept(i));
	}
	
	@Override
	public boolean isValid(int id) {
		return contains(id);
	}
	
	@Override
	public int size() {
		return validIdCount;
	}

	@Override
	public float getInclusionRate() {
		return Math.max(0, Math.min(1, ((float)size()) / allElementCount));
	}
	
	@Override
	public String toString() {
		return this.getClass().getSimpleName() + " with "+getInclusionRate()+" valid ids.";
	}
	
	@Override
	public MutableVertexFilter clone() {
		return new MutableVertexFilter(((RoaringBitmap)this).clone(), allElementCount);
	}
}
