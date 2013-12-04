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
import burlap.behavior.singleagent.planning.commonpolicies.EpsilonGreedy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.VisualActionObserver;

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
				maxQ = Math.max(maxQ, qv.q);
			}
			System.out.println("");
		}
		
		double maxR = Double.NEGATIVE_INFINITY;
		StateHashTuple maxRsht = null;
		GroundedAction maxRga = null;
		System.out.println("Experienced Rewards");
		for (StateHashTuple sht : pastExperience.keySet()) {
			printRmaxDebugPos(sht);
			for (GroundedAction ga : pastExperience.get(sht).getEstRewards().keySet()) {
				System.out.printf("%.2f ", pastExperience.get(sht).getEstRewards().get(ga));
				//maxR = Math.max(maxR, pastExperience.get(sht).getEstRewards().get(ga));
				if (maxR < pastExperience.get(sht).getEstRewards().get(ga)) {
					maxR = pastExperience.get(sht).getEstRewards().get(ga);
					maxRsht = sht;
					maxRga = ga;
				}
			}
			System.out.println("");
		}
		System.out.println("Max Q: " + maxQ + "    " + "max R: " + maxR);
		printRmaxDebugPos(maxRsht);
		System.out.println("action " + maxRga);
		System.out.println("Last discounted return: " + 
				getLastLearningEpisode().getDiscountedReturn(this.gamma));
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
			if(gas.isEmpty()){
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
		if(this.tf.isTerminal(s.s)){
			return 0.;
		}
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
		EpisodeAnalysis ea		= new EpisodeAnalysis(initialState);
		StateHashTuple curState	= this.stateHash(initialState);
		eStepCounter			= 0;
		
		while(!tf.isTerminal(curState.s) && eStepCounter < maxEpisodeSize){
			GroundedAction action = learningPolicy.getAction(curState.s);
			StateHashTuple nextState = this.stateHash(action.executeIn(curState.s));
			
			double r = rf.reward(curState.s, action, nextState.s);
			double discount = this.gamma;
			eStepCounter++;
			//System.out.println(eStepCounter);

			ea.recordTransitionTo(nextState.s, action, r);
			
			if (!pastExperience.containsKey(curState)) {
				pastExperience.put(curState, new RmaxMemoryNode(m));
			}			
			RmaxMemoryNode memoryNode = pastExperience.get(curState);
			if (!memoryNode.hasEnoughExperience(action)) {
				memoryNode.addExperience(action,nextState,r);
			}
			
			if (memoryNode.hasEnoughExperience(action) && memoryNode.runVI(action)) {
				memoryNode.updateEstimations(action);
				RmaxMemoryNode					node;
				Map<StateHashTuple, Double>		transitionDist;
				double sum_t_q;
				// update q values
				double qEps = 0.0001;
				//for (int i = 0; i < 1; ++i) {
				double absDelQ;
				int i=0;
				do {
					i++;
					absDelQ = Double.NEGATIVE_INFINITY;
					for (StateHashTuple state: pastExperience.keySet()) { // s
						node = pastExperience.get(state);

						for(GroundedAction a : node.getGroundedActions()) { // a
//							if(this.tf.isTerminal(state.s)){
//								this.getQ(state, a).q = 0;
//								continue;
//							}
							if (node.hasEnoughExperience(a)) {
								sum_t_q = 0.;
								transitionDist = node.getEstTransitionDist(a);
								for (StateHashTuple s_prime : transitionDist.keySet()) { // s'
									
									sum_t_q += transitionDist.get(s_prime) * this.getMaxQ(s_prime);
								}
								sum_t_q *= discount;
								sum_t_q += node.getEstReward(a);
								absDelQ = Math.max(absDelQ,
										Math.abs(this.getQ(state, a).q - sum_t_q));
								this.getQ(state, a).q = sum_t_q;
							}
						}
					}
				} while (absDelQ > qEps);
				System.out.println("VI iterations: "+i);
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
