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
	private int													m;

	public RmaxMemoryNode(int m) {
		this.pastRewards = new HashMap<GroundedAction, Double>();
		this.pastSAS = new HashMap<GroundedAction, Map<StateHashTuple,Integer>>();
		this.pastSA = new HashMap<GroundedAction, Integer>();
		this.updatedActions = new HashSet<GroundedAction>();
		this.estTransition = new HashMap<GroundedAction, Map<StateHashTuple, Double>>();
		this.estRewards = new HashMap<GroundedAction, Double>();
		this.m = m;
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
	
	public boolean runVI(GroundedAction ga) {
		//return true;
		return !this.updatedActions.contains(ga);
	}
	
	public void addExperience(GroundedAction action, StateHashTuple resultState, double reward) {
		//Update rewards
		double r;
		if (this.pastRewards.containsKey(action)) {
			r = this.pastRewards.get(action);
		} else {
			r = 0.;
		}
		this.pastRewards.put(action, r+reward);
		
		//Update state,action pair
		int nSA;
		if (this.pastSA.containsKey(action)) {
			nSA = this.pastSA.get(action);
		} else {
			nSA = 0;
		}
		this.pastSA.put(action, nSA+1);
		
		//Update (state,action,result state)
		Map<StateHashTuple, Integer> nSASMap;
		if (this.pastSAS.containsKey(action)) {
			nSASMap = this.pastSAS.get(action);
		} else {
			nSASMap = new HashMap<StateHashTuple, Integer>();
		}
		int nSAS;
		if (nSASMap.containsKey(resultState)) {
			nSAS = nSASMap.get(resultState);
		} else {
			nSAS = 0;
		}
		nSASMap.put(resultState, nSAS+1);
		this.pastSAS.put(action, nSASMap);
	}
	
	public boolean hasEnoughExperience(GroundedAction action) {
		if (!this.pastSA.containsKey(action)) {
			return false;
		} else {
			return (this.pastSA.get(action) >= m);
		}
	}
	
	public void updateEstimations(GroundedAction action) {
		if (this.updatedActions.contains(action)) return;
		
		this.updatedActions.add(action);
		double totalReward = this.pastRewards.get(action);
		this.estRewards.put(action, totalReward/(double)m);
		
		Map<StateHashTuple, Integer> nSASMap = this.pastSAS.get(action);
		Map<StateHashTuple,Double> transitionMap = new HashMap<StateHashTuple, Double>();
		int nSAS, nSA;		
		for (StateHashTuple sh: nSASMap.keySet()) {
			if (this.pastSAS.containsKey(action)) {
				nSAS = this.pastSAS.get(action).get(sh);
			} else {
				nSAS = 0;
			}
			nSA = this.pastSA.get(action);
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
		for (GroundedAction a: this.estTransition.keySet()) {
			System.out.println(a.hashCode());
		}
	}
}
