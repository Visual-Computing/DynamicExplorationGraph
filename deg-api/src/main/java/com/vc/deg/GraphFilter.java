package com.vc.deg;

/**
 * A filter to check labels used in a search or exploration task
 * 
 * @author Nico Hezel
 */
public interface GraphFilter {

	/**
	 * Is the label valid
	 * 
	 * @param label
	 * @return
	 */
	public boolean isValid(int label);
	
	/**
	 * Number of valid labels
	 * 
	 * @return
	 */
	public int size();
	
	
	/**
	 * This filter always return true when asked for validity
	 * 
	 * @author Nico Hezel
	 */
	public static class AllValidFilter implements GraphFilter {
		
		protected final int size;
		
		public AllValidFilter(int size) {
			this.size = size;
		}
		
		@Override
		public int size() {
			return size;
		}
		
		@Override
		public boolean isValid(int label) {
			return true;
		}
	};
}
