package com.vc.deg.search;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;

import com.vc.deg.data.DataRepository;
import com.vc.deg.data.FeatureSpace;
import com.vc.deg.graph.EvenRegularWeightedUndirectedGraph;

/**
 * 
 * @author Nico Hezel
 *
 */
public class GraphSearch {

	public static <T> TreeSet<ObjectDistance<T>> search(T query, int k, float eps, int[] forbiddenIds, int[] entryPoints, EvenRegularWeightedUndirectedGraph graph, DataRepository<T> objectRepository) {
		final FeatureSpace<T> featureSpace = objectRepository.getFeatureSpace();
		
		// list of checked ids
		final Set<Integer> C = new HashSet<>(forbiddenIds.length + entryPoints.length);
		for (int id : forbiddenIds)
			C.add(id);
		for (int id : entryPoints)
			C.add(id);
		
		// items to traverse, start with the initial node
		final PriorityQueue<ObjectDistance<T>> S = new PriorityQueue<>();
		for (int id : entryPoints) {
			T obj = objectRepository.get(id);
			S.add(new ObjectDistance<>(id, obj, featureSpace.computeDistance(query, obj)));
		}

		// result set
		final TreeSet<ObjectDistance<T>> R = new TreeSet<>(S);

		// search radius
		float r = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(S.size() > 0) {
			final ObjectDistance<T> s = S.poll();

			// max distance reached
			if(s.dist > r * (1 + eps))
				break;

			// traverse never seen nodes
			for(int neighborId : graph.getConnectedNodeIds(s.id).toArray()) {

				if(C.add(neighborId) == false) {
					final T n = objectRepository.get(neighborId);
					final float nDist = featureSpace.computeDistance(query, n);

					// follow this node further
					if(nDist <= r * (1 + eps)) {
						final ObjectDistance<T> candidate = new ObjectDistance<T>(neighborId, n, nDist);
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
	public static class ObjectDistance<T> implements Comparable<ObjectDistance<T>> {
		
		public final int id;
		public final T obj;
		public final float dist;
		
		public ObjectDistance(int id, T obj, float dist) {
			this.id = id;
			this.obj = obj;
			this.dist = dist;
		}

		@Override
		public int compareTo(ObjectDistance<T> o) {
			int cmp = Float.compare(-dist, -o.dist);
			if(cmp == 0)
				cmp = Integer.compare(obj.hashCode(), o.obj.hashCode());
			return cmp;
		}
	}
}
