package com.vc.deg.ref.search;

import java.util.Comparator;

/**
 * Natural order is ascending by distance.
 * 
 * @author Nico Hezel
 *
 */
public class ObjectDistance implements Comparable<ObjectDistance> {
	
	protected final int label;
	protected final float distance;
	
	public ObjectDistance(int label, float distance) {
		this.label = label;
		this.distance = distance;
	}
	
	public int getLabel() {
		return label;
	}

	public float getDistance() {
		return distance;
	}
	
	@Override
	public String toString() {
		return "label:"+label+", distance:"+distance;
	}
	
	@Override
	public int compareTo(ObjectDistance o) {
		int cmp = Float.compare(getDistance(), o.getDistance());
        if (cmp == 0)
        	cmp = Integer.compare(getLabel(), o.getLabel());
        return cmp;
	}	
	
	/**
	 * Order in ascending order using the index
	 *
	 * @return
	 */
	public static Comparator<ObjectDistance> ascByIndex() {
		return Comparator.comparingInt(ObjectDistance::getLabel).thenComparingDouble(ObjectDistance::getDistance);
	}

	/**
	 * Order in descending order using the index
	 * 
	 * @return
	 */
	public static Comparator<ObjectDistance> descByIndex() {
		return ascByIndex().reversed();
	}
}