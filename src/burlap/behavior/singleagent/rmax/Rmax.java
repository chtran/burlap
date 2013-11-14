package burlap.behavior.singleagent.rmax;

import java.util.LinkedList;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;

/**
 * Implementation of the R-MAX PAC-MDP learning algorithm.
 * Reference:
 * Asmuth, Littman, Zinkov.
 * "Potential-based Shaping in Model-based Reinforcement Learning"
 * In Proceedings of AAAI Conference on Artificial Intelligence
 * 2008
 */
public class Rmax extends OOMDPPlanner implements LearningAgent {

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

	/**
	 * Initialize Rmax()
	 * @param domain the domain in which to learn
	 * @param rf the reward function
	 * @param tf the terminal function
	 * @param gamma the discount factor
	 */
	public Rmax(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma, StateHashFactory hashingFactory) {
		this.plannerInit(domain, rf, tf, gamma, hashingFactory);
		numEpisodesToStore = 1;
		episodeHistory = new LinkedList<EpisodeAnalysis>();
		numEpisodesForPlanning = 1;
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
	public EpisodeAnalysis runLearningEpisodeFrom(State initialState) {
		// TODO Auto-generated method stub
		return null;
	}
	
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
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
}
