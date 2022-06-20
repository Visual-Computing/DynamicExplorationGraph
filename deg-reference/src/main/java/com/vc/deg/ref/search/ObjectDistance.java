package com.vc.deg.ref.search;

import com.vc.deg.SearchResult.SearchEntry;
import com.vc.deg.graph.NodeView;

public class ObjectDistance implements SearchEntry {
	
	protected final int nodeId;
	protected final NodeView nodeData;
	protected final float distance;
	
	public ObjectDistance(int nodeId, NodeView nodeData, float distance) {
		this.nodeId = nodeId;
		this.nodeData = nodeData;
		this.distance = distance;
	}
	
	public int getNodeId() {
		return nodeId;
	}

	@Override
	public int getLabel() {
		return nodeData.getLabel();
	}

	@Override
	public float getDistance() {
		return distance;
	}
}