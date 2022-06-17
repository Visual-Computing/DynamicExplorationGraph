package com.vc.deg.ref.search;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.NodeView;
import com.vc.deg.ref.graph.EvenRegularWeightedUndirectedGraph;

/**
 * 
 * @author Nico Hezel
 *
 */
public class GraphSearch {

	public static <T> TreeSet<ObjectDistance> search(FeatureVector query, int k, float eps, int[] forbiddenIds, int[] entryPoints, EvenRegularWeightedUndirectedGraph graph) {
		final FeatureSpace featureSpace = graph.getFeatureSpace();
		
		// list of checked ids
		final Set<Integer> C = new HashSet<>(forbiddenIds.length + entryPoints.length);
		for (int id : forbiddenIds)
			C.add(id);
		for (int id : entryPoints)
			C.add(id);
		
		// items to traverse, start with the initial node
		final PriorityQueue<ObjectDistance> S = new PriorityQueue<>();
		for (int id : entryPoints) {
			final NodeView obj = graph.getNode(id);
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
			for(int neighborId : graph.getConnectedNodeIds(s.id).toArray()) {

				if(C.add(neighborId) == false) {
					final NodeView n = graph.getNode(neighborId);
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
