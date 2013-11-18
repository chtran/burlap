package burlap.behavior.singleagent.rmax;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import javax.management.RuntimeErrorException;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.QValue;
import burlap.behavior.singleagent.ValueFunctionInitialization;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearningStateNode;
import burlap.behavior.singleagent.options.Option;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

/**
 * Implementation of the R-MAX PAC-MDP learning algorithm.
 * Reference:
 * Asmuth, Littman, Zinkov.
 * "Potential-based Shaping in Model-based Reinforcement Learning"
 * In Proceedings of AAAI Conference on Artificial Intelligence
 * 2008
 */
public class Rmax extends OOMDPPlanner implements QComputablePlanner, LearningAgent {
	/**
	 * The tabular mapping from states to Q-values
	 */
	protected Map<StateHashTuple, QLearningStateNode>				qIndex;
	protected Map<StateHashTuple, RmaxMemoryNode>					pastExperience;

	/**
	 * A counter for counting the number of steps in an episode that have been taken thus far
	 */
	protected int													eStepCounter;

	/**
	 * The maximum number of steps that will be taken in an episode before the agent terminates a learning episode
	 */
	protected int													maxEpisodeSize;
	
	/**
	 * Whether options should be decomposed into actions in the returned {@link burlap.behavior.singleagent.EpisodeAnalysis} objects.
	 */
	protected boolean												shouldDecomposeOptions = true;
	
	protected Policy 												learningPolicy;
	
	// For LearningAgent
	/**
	 * The number of the most recent learning episodes to store.
	 */
	protected int													numEpisodesToStore;
	
	/**
	 * the saved previous learning episodes
	 */
	protected LinkedList<EpisodeAnalysis>							episodeHistory;

	// For OOMDPPlanner
	/**
	 * The maximum number of episodes to use for planning
	 */
	protected int													numEpisodesForPlanning;
	protected double												goalReward;
	protected ValueFunctionInitialization							qInitFunction;
	protected int													m;
	/**
	 * Initialize Rmax()
	 * @param domain the domain in which to learn
	 * @param rf the reward function
	 * @param tf the terminal function
	 * @param gamma the discount factor
	 */
	public Rmax(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma,
			StateHashFactory hashingFactory, double goalReward, int maxEpisodeSize, int m) {
		this.plannerInit(domain, rf, tf, gamma, hashingFactory);
		this.pastExperience = new HashMap<StateHashTuple, RmaxMemoryNode>();
		numEpisodesToStore = 1;
		episodeHistory = new LinkedList<EpisodeAnalysis>();
		numEpisodesForPlanning = 1;
		this.qIndex = new HashMap<StateHashTuple, QLearningStateNode>();

		this.goalReward = goalReward;
		double qInitValue;
		if (gamma != 1.) {
			qInitValue = goalReward/(1. - gamma);
		}
		else {
			qInitValue = goalReward;
		}
		this.qInitFunction = new ValueFunctionInitialization.ConstantValueFunctionInitialization(qInitValue);

		this.maxEpisodeSize = maxEpisodeSize;
		
		this.learningPolicy = new GreedyQPolicy(this);
		
		this.m = m;
	}
	
	
	/**
	 * Sets the maximum number of episodes that will be performed when the {@link planFromState(State)} method is called.
	 * @param n the maximum number of episodes that will be performed when the {@link planFromState(State)} method is called.
	 */
	public void setMaximumEpisodesForPlanning(int n) {
		if (n > 0) {
			this.numEpisodesForPlanning = n;
		}
		else {
			this.numEpisodesForPlanning = 1;
		}
	}
	
	// interface LearningAgent requires the following to be implemented:
	@Override
	public EpisodeAnalysis getLastLearningEpisode() {
		return episodeHistory.getLast();
	}
	@Override
	public void setNumEpisodesToStore(int numEps) {
		if (numEps > 0) {
			numEpisodesToStore = numEps;
		}
		else {
			numEpisodesToStore = 1;
		}
	}
	@Override
	public List<EpisodeAnalysis> getAllStoredLearningEpisodes() {
		return episodeHistory;
	}
	
	// abstract class OOMDPPlanner requires the following to be implemented:
	@Override
	public void planFromState(State initialState) {
		int eCount = 0;
		do {
			this.runLearningEpisodeFrom(initialState);
			eCount++;
		} while (eCount < numEpisodesForPlanning);
		
	}
	
	public void printRmaxDebug() {
		double maxQ = Double.NEGATIVE_INFINITY;
		System.out.println("Q-Values");
		for (StateHashTuple sht : qIndex.keySet()) {
			printRmaxDebugPos(sht);
			for (QValue qv : qIndex.get(sht).qEntry) {
				System.out.printf("%.2f ", qv.q);
				//System.out.print(qv.q + " ");
				maxQ = Math.max(maxQ, qv.q);
			}
			System.out.println("");
		}
		
		double maxR = Double.NEGATIVE_INFINITY;
		System.out.println("Experienced Rewards");
		for (StateHashTuple sht : pastExperience.keySet()) {
			printRmaxDebugPos(sht);
			for (GroundedAction ga : pastExperience.get(sht).getEstRewards().keySet()) {
				System.out.printf("%.2f ", pastExperience.get(sht).getEstRewards().get(ga));
				//System.out.print(pastExperience.get(sht).getEstRewards().get(ga) + " ");
				maxR = Math.max(maxR, pastExperience.get(sht).getEstRewards().get(ga));
			}
			System.out.println("");
		}
		System.out.println("Max Q: " + maxQ + "    " + "max R: " + maxR);
	}
	
	public void printRmaxDebugPos(StateHashTuple sht) {
		ObjectInstance agent = sht.s.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
		int x = agent.getDiscValForAttribute(GridWorldDomain.ATTX);
		int y = agent.getDiscValForAttribute(GridWorldDomain.ATTY);
		System.out.print("x = " + x + ", y = " + y + ":  ");
	}
	
	@Override
	public List<QValue> getQs(State s) {
		return this.getQs(this.stateHash(s));
	}

	@Override
	public QValue getQ(State s, GroundedAction a) {
		return this.getQ(this.stateHash(s), a);
	}
	
	/**
	 * Returns the possible Q-values for a given hashed stated.
	 * @param s the hashed state for which to get the Q-values.
	 * @return the possible Q-values for a given hashed stated.
	 */
	protected List<QValue> getQs(StateHashTuple s) {
		QLearningStateNode node = this.getStateNode(s);
		return node.qEntry;
	}


	/**
	 * Returns the Q-value for a given hashed state and action.
	 * @param s the hashed state
	 * @param a the action
	 * @return the Q-value for a given hashed state and action; null is returned if there is not Q-value currently stored.
	 */
	protected QValue getQ(StateHashTuple s, GroundedAction a) {
		QLearningStateNode node = this.getStateNode(s);
		
		if(a.params.length > 0 && !this.domain.isNameDependent()){
			Map<String, String> matching = s.s.getObjectMatchingTo(node.s.s, false);
			a = this.translateAction(a, matching);
		}
		
		for(QValue qv : node.qEntry){
			if(qv.a.equals(a)){
				return qv;
			}
		}
		
		return null; //no action for this state indexed
	}
	
	/**
	 * Returns the {@link QLearningStateNode} object stored for the given hashed state. If no {@link QLearningStateNode} object.
	 * is stored, then it is created and has its Q-value initialize using this objects {@link burlap.behavior.singleagent.ValueFunctionInitialization} data member.
	 * @param s the hashed state for which to get the {@link QLearningStateNode} object
	 * @return the {@link QLearningStateNode} object stored for the given hashed state. If no {@link QLearningStateNode} object.
	 */
	protected QLearningStateNode getStateNode(StateHashTuple s){
		
		QLearningStateNode node = qIndex.get(s);
		
		if(node == null){
			node = new QLearningStateNode(s);
			List<GroundedAction> gas = this.getAllGroundedActions(s.s);
			if(gas.size() == 0){
				gas = this.getAllGroundedActions(s.s);
				throw new RuntimeErrorException(new Error("No possible actions in this state, cannot continue Q-learning"));
			}
			for(GroundedAction ga : gas){
				node.addQValue(ga, qInitFunction.qValue(s.s, ga));
			}
			
			qIndex.put(s, node);
		}
		
		return node;
		
	}
	
	
	
	protected RmaxMemoryNode getMemoryNode(StateHashTuple s){
		return this.pastExperience.get(s);
		
	}
	
	
	
	
	
	/**
	 * Returns the maximum Q-value in the hashed stated.
	 * @param s the state for which to get he maximum Q-value;
	 * @return the maximum Q-value in the hashed stated.
	 */
	protected double getMaxQ(StateHashTuple s){
		List <QValue> qs = this.getQs(s);
		double max = Double.NEGATIVE_INFINITY;
		for(QValue q : qs){
			if(q.q > max){
				max = q.q;
			}
		}
		return max;
	}
	
	
	@Override
	public EpisodeAnalysis runLearningEpisodeFrom(State initialState) {
		
		//this.toggleShouldAnnotateOptionDecomposition(shouldAnnotateOptions);
		
		EpisodeAnalysis ea = new EpisodeAnalysis(initialState);
		
		StateHashTuple curState = this.stateHash(initialState);
		eStepCounter = 0;
		
		//maxQChangeInLastEpisode = 0.;
		
		while(!tf.isTerminal(curState.s) && eStepCounter < maxEpisodeSize){
			//System.out.println(eStepCounter);
			GroundedAction action = learningPolicy.getAction(curState.s);
			//QValue curQ = this.getQ(curState, action);
			
			StateHashTuple nextState = this.stateHash(action.executeIn(curState.s));
			
//			double maxQ = 0.;
//			
//			if(!tf.isTerminal(nextState.s)){
//				maxQ = this.getMaxQ(nextState);
//			}
			
			//manage option specifics
			double r = 0.;
			double discount = this.gamma;
			if(action.action.isPrimitive()){
				r = rf.reward(curState.s, action, nextState.s);
				eStepCounter++;
				ea.recordTransitionTo(nextState.s, action, r);
			}
			else{
				Option o = (Option)action.action;
				r = o.getLastCumulativeReward();
				int n = o.getLastNumSteps();
				discount = Math.pow(this.gamma, n);
				eStepCounter += n;
				if(this.shouldDecomposeOptions){
					ea.appendAndMergeEpisodeAnalysis(o.getLastExecutionResults());
				}
				else{
					ea.recordTransitionTo(nextState.s, action, r);
				}
			}
//			if (r>0) {
//				System.out.println("Found goal");
//			}
			RmaxMemoryNode memoryNode = new RmaxMemoryNode();
			if (pastExperience.containsKey(curState)) {
				memoryNode = pastExperience.get(curState);
			} else {
				pastExperience.put(curState, memoryNode);
			}
			if (!memoryNode.hasEnoughExperience(action,m))
			{
				memoryNode.addExperience(action,nextState,r);
			}
			
			if (memoryNode.hasEnoughExperience(action,m)) {
				memoryNode.updateEstimations(action, m);
				for (int i=0; i<1; i++) {
					for (StateHashTuple state: pastExperience.keySet()) { // s
						RmaxMemoryNode node = pastExperience.get(state); // s
						for(GroundedAction a : node.getGroundedActions()) { // a
							if (node.hasEnoughExperience(a, m))
							{
								double sum_t_q = 0.;
	
								Map<StateHashTuple, Double> transitionDist = node.getEstTransitionDist(a);
								for (StateHashTuple s_prime : transitionDist.keySet()) // s'
								{
									sum_t_q += transitionDist.get(s_prime)* this.getMaxQ(s_prime);
								}
								sum_t_q *= discount;
								sum_t_q += node.getEstReward(a);
								QValue iterQ = this.getQ(state, a);
								iterQ.q = sum_t_q;
								//System.out.println(sum_t_q);
							}
							
						}
					}
				}
			}		
			
			//move on
			curState = nextState;
			
			
		}
		
		if(episodeHistory.size() >= numEpisodesToStore){
			episodeHistory.poll();
		}
		episodeHistory.offer(ea);
		
		return ea;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
}
