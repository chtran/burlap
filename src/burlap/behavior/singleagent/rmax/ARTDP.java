package burlap.behavior.singleagent.rmax;

import java.util.Map;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class ARTDP extends Rmax {

	protected double initial_temp;
	protected double min_temp;
	protected double temp_decay_constant;
	
	public ARTDP(Domain domain, RewardFunction rf, TerminalFunction tf,
			double gamma, StateHashFactory hashingFactory, double goalReward,
			int maxEpisodeSize, int m,
			double initial_temp, double min_temp, double temp_decay_constant) {
		super(domain, rf, tf, gamma, hashingFactory, goalReward, maxEpisodeSize, m);
		set_temp_param(initial_temp, min_temp, temp_decay_constant);
		this.learningPolicy = new BoltzmannQPolicy(this, initial_temp, min_temp, temp_decay_constant);		
	}

	public void set_temp_param(double initial_temp, double min_temp, double temp_decay_constant) {
		this.initial_temp = initial_temp;
		this.min_temp = min_temp;
		this.temp_decay_constant = temp_decay_constant;
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
			eStepCounter++;

			ea.recordTransitionTo(nextState.s, action, r);
			
			if (!pastExperience.containsKey(curState)) {
				pastExperience.put(curState, new RmaxMemoryNode(m));
			}			
			RmaxMemoryNode memoryNode = pastExperience.get(curState);
			memoryNode.addExperience(action,nextState,r);
			
			if (memoryNode.hasEnoughExperience(action)) {
				memoryNode.updateEstimations(action);
				// Update Q values
				double sum_t_q;
				Map<StateHashTuple, Double> transitionDist;
				sum_t_q = 0.;
				transitionDist = memoryNode.getEstTransitionDist(action);
				for (StateHashTuple s_prime : transitionDist.keySet()) {
					sum_t_q += transitionDist.get(s_prime) * this.getMaxQ(s_prime);
				}
				sum_t_q *= this.gamma;
				sum_t_q += memoryNode.getEstReward(action);
				this.getQ(curState.s, action).q = sum_t_q;
			}
			
			//move on
			curState = nextState;
			//Update temperature
			((BoltzmannQPolicy) learningPolicy).decay();
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
