package burlap.behavior.singleagent.rmax;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.singleagent.GroundedAction;

public class RmaxMemoryNode {
	
	private Map<GroundedAction, Double>							pastRewards;
	private Map<GroundedAction, Map<StateHashTuple,Integer>>	pastSAS;
	private Map<GroundedAction, Integer>						pastSA;
	private HashSet<GroundedAction>								updatedActions;
	private Map<GroundedAction, Map<StateHashTuple, Double>>	estTransition;
	private Map<GroundedAction, Double>							estRewards;

	public RmaxMemoryNode() {
		this.pastRewards = new HashMap<GroundedAction, Double>();
		this.pastSAS = new HashMap<GroundedAction, Map<StateHashTuple,Integer>>();
		this.pastSA = new HashMap<GroundedAction, Integer>();
		this.updatedActions = new HashSet<GroundedAction>();
		this.estTransition = new HashMap<GroundedAction, Map<StateHashTuple, Double>>();
		this.estRewards = new HashMap<GroundedAction, Double>();
	}
	
	public Set<GroundedAction> getGroundedActions() {
		return this.pastSA.keySet();
	}
	
	public Map<GroundedAction, Double> getPastRewards() {
		return this.pastRewards;
	}
	
	public Map<GroundedAction, Double> getEstRewards() {
		return this.estRewards;
	}
	
	public void addExperience(GroundedAction action, StateHashTuple resultState, double reward) {
		//Update rewards
		double r = 0.;
		if (pastRewards.containsKey(action)) {
			r = pastRewards.get(action);
		}
		pastRewards.put(action, r+reward);
		
		//Update state,action pair
		int nSA = 0;
		if (pastSA.containsKey(action)) {
			nSA = pastSA.get(action);
		}
		pastSA.put(action, nSA+1);
		
		//Update (state,action,result state)
		Map<StateHashTuple, Integer> nSASMap;
		if (pastSAS.containsKey(action)) {
			nSASMap = pastSAS.get(action);
		} else {
			nSASMap = new HashMap<StateHashTuple, Integer>();
		}
		int nSAS = 0;
		if (nSASMap.containsKey(resultState)) {
			nSAS = nSASMap.get(resultState);
		}
		nSASMap.put(resultState, nSAS+1);
		this.pastSAS.put(action, nSASMap);
	}
	
	public boolean hasEnoughExperience(GroundedAction action, int m) {
		if (!pastSA.containsKey(action)) {
			return false;
		} else {
			return (pastSA.get(action) >= m);
		}
	}
	
	public void updateEstimations(GroundedAction action, int m) {
		if (this.updatedActions.contains(action)) return;
		
		this.updatedActions.add(action);
		double totalReward = this.pastRewards.get(action);
		this.estRewards.put(action, totalReward/(double)m);
		
		Map<StateHashTuple, Integer> nSASMap = pastSAS.get(action);
		Map<StateHashTuple,Double> transitionMap = new HashMap<StateHashTuple, Double>();
		for (StateHashTuple sh: nSASMap.keySet()) {
			int nSAS = 0;
			if (pastSAS.containsKey(action)) {
				nSAS = pastSAS.get(action).get(sh);
			}
			int nSA = pastSA.get(action);
			transitionMap.put(sh, ((double)nSAS)/(double)nSA);
		}
		this.estTransition.put(action, transitionMap);
	}
	
	public double getEstReward(GroundedAction action) {
		return this.estRewards.get(action);
	}
	
	public Map<StateHashTuple, Double> getEstTransitionDist(GroundedAction action) {
		if (!this.estTransition.containsKey(action)) {
			printHashCodes();
		}
		return this.estTransition.get(action);
	}
	
	public double getEstTransitionProb(GroundedAction action, StateHashTuple state) {
		return this.estTransition.get(action).get(state);
	}
	
	public void printHashCodes() {
		System.out.println("estTransition");
		for (GroundedAction a: estTransition.keySet()) {
			System.out.println(a.hashCode());
		}
	}
}
