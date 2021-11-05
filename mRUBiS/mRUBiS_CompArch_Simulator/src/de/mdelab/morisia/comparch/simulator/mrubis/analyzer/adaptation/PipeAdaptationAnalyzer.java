package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.PerformanceStats;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.AdaptationAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Analyzes an adaptation of the pipe and reacts to the adaptation by changing
 * the performance stats in the model as if the adaptation impacts the
 * performance of the system.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class PipeAdaptationAnalyzer implements AdaptationAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(PipeAdaptationAnalyzer.class.getName());

	// key: tenant.uid
	// value: array of filter type names while the filters are ordered by their
	// position in the pipe
	private Map<String, String[]> tenantToPipes;
	private int performanceMonitoringIntervall;

	/**
	 * Constructor.
	 * 
	 * @param performanceMonitoringIntervall
	 *            interval in terms of invocation counts between two performance
	 *            measurements.
	 */
	public PipeAdaptationAnalyzer(int performanceMonitoringIntervall) {
		this.tenantToPipes = new HashMap<String, String[]>();
		this.performanceMonitoringIntervall = performanceMonitoringIntervall;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Pipe Adaptation Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes an adaptation of the pipe and reacts to the adaptation by changing "
				+ "the performance stats in the model as if the adaptation impacts the "
				+ "performance of the system.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {
		if (this.tenantToPipes.size() == 0) {
			this.snapshotPipes(eArchitecture);
		}		
		for (Tenant tenant : eArchitecture.getTenants()) {
			LOGGER.info("Analyzing the tenant \"" + tenant.getName() + "\" ...");
			PipeChangeType pipeChangeType = this.hasPipeChanged(tenant);
			if (pipeChangeType == PipeChangeType.NO_CHANGE) {
				LOGGER.info("  -- Pipe of tenant \"" + tenant.getName()
						+ "\" does not seem to have changed.");
			} else if (pipeChangeType == PipeChangeType.FILTER_ORDER_CHANGE) {
				boolean wellOrderedPipe = this.areFiltersWellOrderd(
						eArchitecture, tenant);
				if (wellOrderedPipe) {
					LOGGER.info("  -- Pipe of the tenant \""
							+ tenant.getName()
							+ "\" seems to have been properly reorderd, which might result in a decrease of the average response time.");
					// reduce average response time by 5%
					this.changeAverageResponseTime(tenant, 0.95);
					// TODO blacklist tenant?
				}
			} else if (pipeChangeType == PipeChangeType.FILTER_ADDED) {
				LOGGER.info("  -- Pipe of the tenant \""
						+ tenant.getName()
						+ "\" seems to have more filters than before, which might result in an increase of the average response time.");
				// identify the added filters
				List<Component> addedFilters = this.getAddedFilters(tenant);
				if (addedFilters.size() > 0) {
					long absoluteIncrease = this
							.getSumOfComputationTime(addedFilters);
					this.changeAverageResponseTime(tenant, absoluteIncrease);
					// log added filters
					String msg = " -- Added filters:";
					for (Component addedFilter : addedFilters) {
						msg = msg + "\n         -- \"" + addedFilter.getUid()
								+ "\"";
					}
					LOGGER.finer(msg);
				} else {
					// increase by 15%
					this.changeAverageResponseTime(tenant, 1.15);
				}
				// TODO blacklist tenant?
			} else if (pipeChangeType == PipeChangeType.FILTER_SKIPPED) {
				LOGGER.info("  -- Pipe of the tenant \""
						+ tenant.getName()
						+ "\" seems to have less filters than before, which might result in a decrease of the average response time.");
				// identify the skipped filters
				List<Component> skippedFilters = this.getSkippedFilters(tenant);
				if (skippedFilters.size() > 0) {
					long absoluteDecrease = this
							.getSumOfComputationTime(skippedFilters) * -1;
					this.changeAverageResponseTime(tenant, absoluteDecrease);
					// log skipped filters
					String msg = " -- Skipped filters:";
					for (Component skippedFilter : skippedFilters) {
						msg = msg + "\n         -- \"" + skippedFilter.getUid()
								+ "\"";
					}
					LOGGER.finer(msg);
				} else {
					// decrease by 20%
					this.changeAverageResponseTime(tenant, 0.80);
				}
				// TODO blacklis tenant?
			}
		}
		// proper time/place to create the snapshot? Should be ok!
		this.snapshotPipes(eArchitecture);
		return 0;
	}

	private void snapshotPipes(Architecture architecture) {
		for (Tenant tenant : architecture.getTenants()) {
			List<Component> filters = MrubisModelQuery.INSTANCE
					.getAllFiltersOfTenant(tenant);
			// snapshot only the pipe if there is at least one filter in the
			// pipe.
			if (filters.size() > 0) {
				String[] filterTypeNames = new String[filters.size()];
				for (int position = 0; position < filters.size(); position++) {
					Component filter = filters.get(position);
					filterTypeNames[position] = filter.getType().getName();
				}
				tenantToPipes.put(tenant.getUid(), filterTypeNames);
			} else {
				String[] filterTypeNames = new String[0];
				tenantToPipes.put(tenant.getUid(), filterTypeNames);
				LOGGER.warning(" -- Snapshot of the empty pipe of filters created for the tenant \""
						+ tenant.getName() + "\".");
			}
		}
	}

	private PipeChangeType hasPipeChanged(Tenant tenant) {
		String[] filterTypeNames = this.tenantToPipes.get(tenant.getUid());
		List<Component> filters = MrubisModelQuery.INSTANCE
				.getAllFiltersOfTenant(tenant);
		if (filters.size() == filterTypeNames.length) {
			for (int position = 0; position < filters.size(); position++) {
				if (!filterTypeNames[position].equals(filters.get(position)
						.getType().getName())) {
					return PipeChangeType.FILTER_ORDER_CHANGE;
				}
			}
			return PipeChangeType.NO_CHANGE;
		} else if (filters.size() > filterTypeNames.length) {
			return PipeChangeType.FILTER_ADDED;
		} else if (filters.size() < filterTypeNames.length) {
			return PipeChangeType.FILTER_SKIPPED;
		} else {
			return PipeChangeType.NO_CHANGE;
		}
	}

	private List<Component> getAddedFilters(Tenant tenant) {
		String[] filterTypeNames = this.tenantToPipes.get(tenant.getUid());
		List<Component> filters = MrubisModelQuery.INSTANCE
				.getAllFiltersOfTenant(tenant);
		List<Component> addedFilters = new LinkedList<>();
		if (filters.size() > filterTypeNames.length) {
			for (Component filter : filters) {
				String filterType = filter.getType().getName();
				boolean match = false;
				for (int i = 0; i < filterTypeNames.length; i++) {
					if (filterType.equals(filterTypeNames[i])) {
						// filter is already part of the snapshot
						match = true;
					}
				}
				if (!match) {
					addedFilters.add(filter);
				}
			}
		}
		return addedFilters;
	}

	private List<Component> getSkippedFilters(Tenant tenant) {
		String[] filterTypeNames = this.tenantToPipes.get(tenant.getUid());
		List<Component> filters = MrubisModelQuery.INSTANCE
				.getAllFiltersOfTenant(tenant);
		List<Component> skippedFilters = new LinkedList<>();
		if (filters.size() < filterTypeNames.length) {
			for (int i = 0; i < filterTypeNames.length; i++) {
				boolean match = false;
				for (Component filter : filters) {
					String filterType = filter.getType().getName();
					if (filterType.equals(filterTypeNames[i])) {
						// filter is already part of the snapshot
						match = true;
					}
				}

				if (!match) {
					// find filter of type filterTypeNames[i] in tenant
					OUTERLOOP: for (ComponentType ct : tenant.getArchitecture()
							.getComponentTypes()) {
						if (ct.getName().equals(filterTypeNames[i])) {
							for (Component c : ct.getInstances()) {
								if (c.getTenant().equals(tenant)) {
									skippedFilters.add(c);
									break OUTERLOOP;
								}
							}
						}
					}
				}
			}
		}
		return skippedFilters;
	}

	private boolean areFiltersWellOrderd(Architecture architecture,
			Tenant tenant) {
		boolean wellOrdered = true;

		List<Component> filters = MrubisModelQuery.INSTANCE
				.getAllFiltersOfTenant(tenant);
		Double[] slopes = new Double[filters.size()];

		// calculate slopes
		for (int i = 0; i < filters.size(); i++) {
			Component filter = filters.get(i);
			double slope = MrubisModelQuery.INSTANCE.getSlopeOfFilter(filter);
			slopes[i] = slope;
		}

		// compare slopes
		for (int j = 0; j < slopes.length - 1; j++) {
			if ((slopes[j] - slopes[j + 1]) < 0) {
				wellOrdered = false;
			}
		}

		return wellOrdered;
	}

	private void changeAverageResponseTime(Tenant tenant,
			long deltaToAverageResponseTime) {
		try {
			// find performance stats
			PerformanceStats targetMethodStats = MrubisModelQuery.INSTANCE
					.getPerformanceStats(tenant);

			long totalTime = targetMethodStats.getTotalTime();
			long invocationCount = targetMethodStats.getInvocationCount();
			long old_avgResponseTime = totalTime / invocationCount;
			long new_avgResponseTime = old_avgResponseTime
					+ deltaToAverageResponseTime;

			long new_invocationCount = invocationCount
					+ this.performanceMonitoringIntervall;
			long new_totalTime = new_invocationCount * new_avgResponseTime;

			targetMethodStats.setTotalTime(new_totalTime);
			targetMethodStats.setInvocationCount(new_invocationCount);

			this.scaleOtherPerformanceStats(tenant, targetMethodStats);
		} catch (Exception e) {
			LOGGER.severe("Simulator cannot change the average response time as after an adaptation: "
					+ e.getMessage());
		}
	}

	private void changeAverageResponseTime(Tenant tenant, double scalar) {

		try {
			// find browseCategoriesInterface
			ProvidedInterface browseCategoriesInterface = MrubisModelQuery.INSTANCE
					.getProvidedInterfaceByName(tenant,
							MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE);
			// find method
			MethodSpecification targetMethod = MrubisModelQuery.INSTANCE
					.findMethodSpecification(browseCategoriesInterface,
							MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE);
			// find performance stats
			PerformanceStats targetMethodStats = MrubisModelQuery.INSTANCE
					.getStatsForMethod(browseCategoriesInterface, targetMethod);

			long totalTime = targetMethodStats.getTotalTime();
			long invocationCount = targetMethodStats.getInvocationCount();
			long old_avgResponseTime = totalTime / invocationCount;
			long new_avgResponseTime = (long) (old_avgResponseTime * scalar);

			long new_invocationCount = invocationCount
					+ this.performanceMonitoringIntervall;
			long new_totalTime = new_invocationCount * new_avgResponseTime;

			targetMethodStats.setTotalTime(new_totalTime);
			targetMethodStats.setInvocationCount(new_invocationCount);

			this.scaleOtherPerformanceStats(tenant, targetMethodStats);
		} catch (Exception e) {
			LOGGER.severe("Simulator cannot change the average response time as after an adaptation: "
					+ e.getMessage());
		}
	}

	/**
	 * scale all other performance stats of this tenant linearly such that the
	 * average response time remains the same
	 * 
	 * @param tenant
	 *            the tenent
	 * @param targetMethodStats
	 *            the performace stats element of the method to be excluded from
	 *            the scaling
	 */
	private void scaleOtherPerformanceStats(Tenant tenant,
			PerformanceStats targetMethodStats) {

		for (Component comp : tenant.getComponents()) {
			for (ProvidedInterface pi : comp.getProvidedInterfaces()) {
				for (PerformanceStats ps : pi.getPerformanceStats()) {
					if (!ps.equals(targetMethodStats)) {
						// only scale stats of methods that are used
						if (ps.getInvocationCount() > 0) {
							long ps_avgResponseTime = ps.getTotalTime()
									/ ps.getInvocationCount();
							ps.setInvocationCount(ps.getInvocationCount()
									+ this.performanceMonitoringIntervall);
							ps.setTotalTime(ps_avgResponseTime
									* ps.getInvocationCount());
						}
					}
				}
			}
		}
	}

	private long getSumOfComputationTime(List<Component> filters) {
		long sum = 0;
		for (Component f : filters) {
			MonitoredProperty c = MrubisModelQuery.INSTANCE
					.getMonitoredPropertyByName(f,
							MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_NAME);
			long fTime = 0;
			if (c != null) {
				fTime = Long.parseLong(c.getValue());
			}
			sum = sum + fTime;
		}
		return sum;
	}

	public enum PipeChangeType {
		FILTER_ORDER_CHANGE, FILTER_ADDED, FILTER_SKIPPED, NO_CHANGE;
	}

}
