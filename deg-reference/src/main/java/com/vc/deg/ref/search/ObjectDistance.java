package com.vc.deg.ref.search;

import java.util.Comparator;

import com.vc.deg.FeatureVector;

/**
 * Natural order is ascending by distance.
 * 
 * @author Nico Hezel
 *
 */
public class ObjectDistance implements Comparable<ObjectDistance> {
	
	protected final int queryId;
	protected final FeatureVector queryFeature;
	
	protected final int objId;
	protected final FeatureVector objFeature;
	
	protected final float distance;
	
	public ObjectDistance(int queryId, FeatureVector queryFeature, int objId, FeatureVector objFeature, float distance) {
		this.queryId = queryId;
		this.queryFeature = queryFeature;
		this.objId = objId;
		this.objFeature = objFeature;
		this.distance = distance;
	}
	

	public int getQueryId() {
		return queryId;
	}
	
	public FeatureVector getQueryFeature() {
		return queryFeature;
	}
	
	public int getObjId() {
		return objId;
	}
	
	public FeatureVector getObjFeature() {
		return objFeature;
	}

	public float getDistance() {
		return distance;
	}
	
	@Override
	public String toString() {
		return "label:"+objId+", distance:"+distance;
	}
	
	@Override
	public int compareTo(ObjectDistance o) {
		int cmp = Float.compare(getDistance(), o.getDistance());
        if (cmp == 0)
        	cmp = Integer.compare(getObjId(), o.getObjId());
        return cmp;
	}	
	
	/**
	 * Order in ascending order using the index
	 *
	 * @return
	 */
	public static Comparator<ObjectDistance> ascByIndex() {
		return Comparator.comparingInt(ObjectDistance::getObjId).thenComparingDouble(ObjectDistance::getDistance);
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