package com.vc.deg.impl.search;

import java.util.PriorityQueue;
import java.util.TreeSet;

import com.koloboke.collect.IntCursor;
import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.NodeView;
import com.vc.deg.impl.graph.WeightedUndirectedGraph;

/**
 * 
 * @author Neiko
 *
 */
public class GraphSearch {

	public static TreeSet<ObjectDistance> search(FeatureVector query, int k, float eps, int[] forbiddenIds, int[] entryPoints, WeightedUndirectedGraph graph) {
		final FeatureSpace featureSpace = graph.getFeatureSpace();
		
		// list of checked ids
		final IntSet C = HashIntSets.newMutableSet(forbiddenIds, forbiddenIds.length + entryPoints.length);
		for (int id : entryPoints)
			C.add(id);
		
		// items to traverse, start with the initial node
		final PriorityQueue<ObjectDistance> S = new PriorityQueue<>();
		for (int id : entryPoints) {
			final NodeView obj = graph.getNodeView(id);
			S.add(new ObjectDistance(id, obj, featureSpace.computeDistance(query, obj.getFeature())));
		}

		// result set
		final TreeSet<ObjectDistance> R = new TreeSet<>(S);

		// search radius
		float r = Float.MAX_VALUE;

		// iterate as long as good elements are in S
		while(S.size() > 0) {
			final ObjectDistance s = S.poll();

			// max distance reached
			if(s.dist > r * (1 + eps))
				break;

			// traverse never seen nodes
			final IntCursor topListCursor = graph.getEdgeIds(s.id).cursor();
			while(topListCursor.moveNext()) {
				final int neighborId = topListCursor.elem();

				if(C.add(neighborId) == false) {
					final NodeView n = graph.getNodeView(neighborId);
					final float nDist = featureSpace.computeDistance(query, n.getFeature());

					// follow this node further
					if(nDist <= r * (1 + eps)) {
						final ObjectDistance candidate = new ObjectDistance(neighborId, n, nDist);
						S.add(candidate);

						// remember the node
						if(nDist < r) {
							R.add(candidate);
							if(R.size() > k) {
								R.pollLast();
								r = R.last().dist;
							}							
						}
					}					
				}
			}
		}
		
		return R;
	}
		
	/**
	 * 
	 * @author Neiko
	 *
	 * @param <T>
	 */
	public static class ObjectDistance implements Comparable<ObjectDistance> {
		
		public final int id;
		public final NodeView obj;
		public final float dist;
		
		public ObjectDistance(int id, NodeView obj, float dist) {
			this.id = id;
			this.obj = obj;
			this.dist = dist;
		}

		@Override
		public int compareTo(ObjectDistance o) {
			int cmp = Float.compare(-dist, -o.dist);
			if(cmp == 0)
				cmp = Integer.compare(obj.hashCode(), o.obj.hashCode());
			return cmp;
		}
	}
}
