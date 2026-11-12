// ============================================================================
// MMU NEST (Near-Memory Translation with RLDRAM Cuckoo Hashing)
// ============================================================================
//
// Architecture Overview:
// ~~~~~~~~~~~~~~~~~~~~~~
//   +--------+     +------------+     +------------+
//   |  Core  | --> |  MMU NEST  | --> |   Memory   |
//   +--------+     +------------+     +------------+
//                        |
//          +-------------+-------------+
//          |             |             |
//     +----v----+   +----v----+   +----v----+
//     | HW TLBs |   |  NEST   |   |   PTW   |
//     | (SRAM)  |   | (RLDRAM)|   | (DRAM)  |
//     +---------+   +---------+   +---------+
//
// Translation Flow:
// ~~~~~~~~~~~~~~~~~
//   1. Lookup HW TLB hierarchy (L1 iTLB, L1 dTLB, L2 TLB)
//   2. On HW TLB miss: Parallel lookup in NEST 4-bank Cuckoo Table (10 ns)
//   3. On NEST miss: Fall back to Page Table Walk in main DRAM
//   4. On PTW completion: Populate HW TLBs and NEST Cuckoo Table
// ============================================================================

#include "mmu_nest.h"
#include "mmu_base.h"
#include "tlb.h"
#include "memory_manager.h"
#include "cache_cntlr.h"
#include "subsecond_time.h"
#include "fixed_types.h"
#include "pagetable_factory.h"
#include "core.h"
#include "thread.h"
#include "mimicos.h"
#include "instruction.h"
#include "misc/exception_handler_base.h"
#include "sniper_space_exception_handler.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "filter_factory.h"

// #define DEBUG_MMU  // Uncomment for verbose debug logging

using namespace std;

namespace ParametricDramDirectoryMSI
{
	static String cfgStringDefault(const String &key, const String &def)
	{
		return Sim()->getCfg()->hasKey(key) ? Sim()->getCfg()->getString(key) : def;
	}

	static int cfgIntDefault(const String &key, int def)
	{
		return Sim()->getCfg()->hasKey(key) ? Sim()->getCfg()->getInt(key) : def;
	}

	NestShared *NestShared::s_instance = nullptr;
	std::mutex NestShared::s_init_lock;

	NestShared* NestShared::get()
	{
		std::lock_guard<std::mutex> guard(s_init_lock);
		if (!s_instance)
			s_instance = new NestShared();
		return s_instance;
	}

	NestShared::NestShared()
	{
		String mode = cfgStringDefault("perf_model/nest/mode", "cuckoo");
		if (mode == "off")        m_mode = Mode::Off;
		else if (mode == "ideal") m_mode = Mode::Ideal;
		else                      m_mode = Mode::Cuckoo;

		m_latency = SubsecondTime::NS(cfgIntDefault("perf_model/nest/latency_ns", 10));
		int max_kicks = cfgIntDefault("perf_model/nest/max_kicks", 8);

		m_num_page_sizes = Sim()->getCfg()->getInt("perf_model/nest/page_sizes");
		m_ideal.resize(m_num_page_sizes);

		for (int i = 0; i < m_num_page_sizes; i++)
		{
			int page_size = Sim()->getCfg()->getIntArray("perf_model/nest/page_size_list", i);
			int entries   = Sim()->getCfg()->getIntArray("perf_model/nest/entries", i);
			m_page_sizes.push_back(page_size);
			m_tables.push_back(m_mode == Mode::Cuckoo
			                   ? new NestCuckooTable(page_size, entries, max_kicks)
			                   : nullptr);

			std::cout << "[NEST] mode=" << mode << " page_size=" << page_size
			          << " entries=" << entries << " latency_ns=" << m_latency.getNS() << std::endl;
		}
	}

	bool NestShared::lookup(UInt16 asid, IntPtr address, IntPtr &ppn, int &page_size)
	{
		std::lock_guard<std::mutex> guard(m_lock);
		for (int i = 0; i < m_num_page_sizes; i++)
		{
			int ps = m_page_sizes[i];
			IntPtr vpn = address >> ps;
			bool hit = false;

			if (m_mode == Mode::Ideal)
			{
				auto it = m_ideal[i].find(idealKey(asid, vpn));
				if (it != m_ideal[i].end()) { ppn = it->second; hit = true; }
			}
			else
			{
				hit = m_tables[i]->lookup(asid, vpn, ppn);
			}

			if (hit) { page_size = ps; return true; }
		}
		return false;
	}

	void NestShared::insert(UInt16 asid, IntPtr address, IntPtr ppn, int page_size,
	                        size_t &kicks, bool &evicted)
	{
		std::lock_guard<std::mutex> guard(m_lock);
		kicks = 0;
		evicted = false;
		for (int i = 0; i < m_num_page_sizes; i++)
		{
			if (m_page_sizes[i] != page_size)
				continue;
			IntPtr vpn = address >> page_size;
			if (m_mode == Mode::Ideal)
				m_ideal[i][idealKey(asid, vpn)] = ppn;
			else
				evicted = m_tables[i]->insert(asid, vpn, ppn, &kicks);
			return;
		}
	}


// ============================================================================
// Construction / Destruction
// ============================================================================

    /**
     * @brief Construct the POM-TLB MMU.
     *
     * Initializes an MMU with a Part-of-Memory TLB that stores translations
     * in main memory. This provides much larger translation coverage than
     * hardware TLBs, at the cost of DRAM access latency.
     *
     * @param _core           Pointer to the core this MMU belongs to
     * @param _memory_manager Memory manager for cache access
     * @param _shmem_perf_model Performance model for timing
     * @param _name           Configuration name prefix
     * @param _nested_mmu     Optional nested MMU for virtualization
     */
	MemoryManagementUnitNEST::MemoryManagementUnitNEST(Core *_core, 
	                                                       MemoryManagerBase *_memory_manager, 
	                                                       ShmemPerfModel *_shmem_perf_model, 
	                                                       String _name, 
	                                                       MemoryManagementUnitBase *_nested_mmu)
	: MemoryManagementUnitBase(_core, _memory_manager, _shmem_perf_model, _name, _nested_mmu),
	  memory_manager(_memory_manager),
	  m_nest(nullptr),
	  m_trace_file(nullptr)
	{
		// SimLog for debug output
		mmu_nest_log = new SimLog("MMU_NEST", core->getId(), DEBUG_MMU_POMTLB);

		instantiatePageTableWalker();  // Creates PWC filter and PTW MSHRs
		instantiateTLBSubsystem();     // Creates HW TLB hierarchy and POM-TLBs
		registerMMUStats();            // Registers statistics with Sniper
	}

    /**
     * @brief Destroy the POM-TLB MMU.
     *
     * Cleans up all allocated resources including TLBs, POM-TLBs, and walkers.
     */
	MemoryManagementUnitNEST::~MemoryManagementUnitNEST()
	{
		delete mmu_nest_log;
		delete tlb_subsystem;
		delete pt_walkers;
		delete[] translation_stats.tlb_latency_per_level;
		delete[] translation_stats.tlb_hit_page_sizes;
		if (m_trace_file)
			fclose(m_trace_file);
	}

// ============================================================================
// Initialization Methods
// ============================================================================

    /**
     * @brief Initialize the Page Table Walker and Page Walk Cache.
     *
     * Creates the PWC filter for caching intermediate page table entries
     * and the MSHR structure for tracking concurrent page table walks.
     */
	void MemoryManagementUnitNEST::instantiatePageTableWalker()
	{
		String mimicos_name = Sim()->getMimicOS()->getName();
		String page_table_type = Sim()->getCfg()->getString("perf_model/"+mimicos_name+"/page_table_type");
		String page_table_name = Sim()->getCfg()->getString("perf_model/"+mimicos_name+"/page_table_name");

		// Create Page Walk Cache filter
		String filter_type = Sim()->getCfg()->getString("perf_model/" + name + "/ptw_filter_type");
		ptw_filter = FilterPTWFactory::createFilterPTWBase(filter_type, name, core);

		// Create MSHRs for N concurrent page table walks
		pt_walkers = new MSHR(Sim()->getCfg()->getInt("perf_model/"+name+"/page_table_walkers"));
	}

    /**
     * @brief Instantiate metadata table (not used in POM-TLB variant).
     */
	void MemoryManagementUnitNEST::instantiateMetadataTable()
	{
		// No metadata table for this MMU variant
	}

    /**
     * @brief Initialize the TLB hierarchy and POM-TLBs.
     *
     * Creates two components:
     * 1. Hardware TLB hierarchy (L1 iTLB, L1 dTLB, L2 TLB, etc.)
     * 2. Part-of-Memory TLBs - one per page size
     *
     * Each POM-TLB is allocated in physical memory and configured with:
     * - Number of entries and associativity from config
     * - Entry size calculated from tag bits + 52 bits for PPN
     * - Base address stored in software_tlb_base_register[]
     */
	void MemoryManagementUnitNEST::instantiateTLBSubsystem()
	{
		// Create standard hardware TLB hierarchy
		tlb_subsystem = new TLBHierarchy(name, core, memory_manager, shmem_perf_model);
		m_nest = NestShared::get();

		if (Sim()->getCfg()->getBoolDefault("perf_model/nest/trace_enabled", false))
		{
			String prefix = cfgStringDefault("perf_model/nest/trace_prefix", "nest_trace");
			String path = Sim()->getConfig()->formatOutputFileName(
				prefix + "_core" + itostr(core->getId()) + ".bin");
			m_trace_file = fopen(path.c_str(), "wb");
		}
	}

// ============================================================================
// Statistics Registration
// ============================================================================

    /**
     * @brief Register all MMU statistics with Sniper's statistics framework.
     *
     * Statistics include:
     * - page_faults: Pages not present in memory
     * - total_table_walk_latency: Cumulative PTW time
     * - total_fault_latency: Time handling page faults
     * - total_tlb_latency: Cumulative hardware TLB lookup time
     * - total_translation_latency: End-to-end translation time
     * - total_software_tlb_latency: Time spent in POM-TLB lookups
     * - total_walk_delay_latency: Time waiting for PTW MSHRs
     */
	void MemoryManagementUnitNEST::registerMMUStats()
	{
		bzero(&translation_stats, sizeof(translation_stats));

		registerStatsMetric(name, core->getId(), "page_faults", &translation_stats.page_faults);
		registerStatsMetric(name, core->getId(), "total_table_walk_latency", &translation_stats.total_walk_latency);
		registerStatsMetric(name, core->getId(), "total_fault_latency", &translation_stats.total_fault_latency);
		registerStatsMetric(name, core->getId(), "total_tlb_latency", &translation_stats.total_tlb_latency);
		registerStatsMetric(name, core->getId(), "total_translation_latency", &translation_stats.total_translation_latency);
		registerStatsMetric(name, core->getId(), "total_software_tlb_latency", &translation_stats.software_tlb_latency);
		registerStatsMetric(name, core->getId(), "total_walk_delay_latency", &translation_stats.total_walk_delay_latency);
		registerStatsMetric(name, core->getId(), "nest_lookups", &translation_stats.nest_lookups);
		registerStatsMetric(name, core->getId(), "nest_hits", &translation_stats.nest_hits);
		registerStatsMetric(name, core->getId(), "nest_misses", &translation_stats.nest_misses);
		registerStatsMetric(name, core->getId(), "nest_insertions", &translation_stats.nest_insertions);
		registerStatsMetric(name, core->getId(), "nest_kicks", &translation_stats.nest_kicks);
		registerStatsMetric(name, core->getId(), "nest_evictions", &translation_stats.nest_evictions);
		registerStatsMetric(name, core->getId(), "num_translations", &translation_stats.num_translations);
		registerStatsMetric(name, core->getId(), "page_table_walks", &translation_stats.page_table_walks);
		registerStatsMetric(name, core->getId(), "translations_with_fault", &translation_stats.translations_with_fault);

		// Per-level TLB latency statistics
		translation_stats.tlb_latency_per_level = new SubsecondTime[tlb_subsystem->getTLBSubsystem().size()];
		for (UInt32 i = 0; i < tlb_subsystem->getTLBSubsystem().size(); i++)
			registerStatsMetric(name, core->getId(), "tlb_latency_" + itostr(i), &translation_stats.tlb_latency_per_level[i]);
	}

// ============================================================================
// Address Translation - Core MMU Operation
// ============================================================================

    /**
     * @brief Perform virtual-to-physical address translation with POM-TLB.
     *
     * This is the main entry point for address translation. The POM-TLB MMU
     * extends the standard translation flow with a software-managed TLB layer.
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │                  POM-TLB TRANSLATION FLOW                            │
     * ├─────────────────────────────────────────────────────────────────────┤
     * │                                                                      │
     * │   Virtual Addr ──► HW TLB Lookup ──┬──► HIT ──► Physical Address     │
     * │                                    │                                 │
     * │                                    └──► MISS                         │
     * │                                          │                           │
     * │                                          ▼                           │
     * │                                    POM-TLB Lookup                    │
     * │                                    (parallel for each page size)     │
     * │                                          │                           │
     * │                              +-----------+----------+                │
     * │                              ▼                      ▼                │
     * │                            HIT                    MISS               │
     * │                              │                      │                │
     * │                              │                      ▼                │
     * │                              │             Page Table Walk           │
     * │                              │                      │                │
     * │                              │                      ▼                │
     * │                              │           Allocate in POM-TLB         │
     * │                              │                      │                │
     * │                              +----------+-----------+                │
     * │                                         │                            │
     * │                                         ▼                            │
     * │                              Allocate in HW TLBs                     │
     * │                                         │                            │
     * │                                         ▼                            │
     * │                                  Physical Address                    │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * @param eip          Instruction pointer causing this access
     * @param address      Virtual address to translate
     * @param instruction  True if this is an instruction fetch
     * @param lock         Cache coherence lock signal
     * @param modeled      True to model timing
     * @param count        True to update statistics
     * @return Physical address
     */
	IntPtr MemoryManagementUnitNEST::performAddressTranslation(IntPtr eip,
	                                                             IntPtr address,
	                                                             bool instruction,
	                                                             Core::lock_signal_t lock,
	                                                             bool modeled,
	                                                             bool count)
	{
		// Track DRAM accesses for power/performance analysis
		dram_accesses_during_last_walk = 0;

		mmu_nest_log->debug("[MMU:NEST] ---- Starting address translation for virtual address: %lx ---- at time %lu",
			address, shmem_perf_model->getElapsedTime(ShmemPerfModel::_USER_THREAD).getNS());

		SubsecondTime time = shmem_perf_model->getElapsedTime(ShmemPerfModel::_USER_THREAD);

		// Cleanup completed PTW entries from MSHR
		pt_walkers->removeCompletedEntries(time);

		if (count)
			translation_stats.num_translations++;

		// Get reference to hardware TLB hierarchy
		TLBSubsystem tlbs = tlb_subsystem->getTLBSubsystem();

		// ====================================================================
		// PHASE 1: Hardware TLB Lookup
		// ====================================================================

		bool tlb_hit   = false;
		TLB *hit_tlb   = NULL;
		CacheBlockInfo *tlb_block_info_hit = NULL; 
		CacheBlockInfo *tlb_block_info     = NULL; 
		int hit_level  = -1;

		int page_size_result = -1;
		IntPtr ppn_result    = 0;

		// Search hardware TLB hierarchy
		for (UInt32 i = 0; i < tlbs.size(); i++)
		{
			mmu_nest_log->debug("[MMU] Searching TLB at level: %u", i);
			for (UInt32 j = 0; j < tlbs[i].size(); j++)
			{
				bool tlb_stores_instructions = ((tlbs[i][j]->getType() == TLBtype::Instruction) ||
				                                 (tlbs[i][j]->getType() == TLBtype::Unified));

				if (tlb_stores_instructions && instruction)
				{
					tlb_block_info = tlbs[i][j]->lookup(address, time, count, lock, eip, modeled, count, NULL);
					if (tlb_block_info != NULL)
					{
						tlb_block_info_hit = tlb_block_info;
						hit_tlb = tlbs[i][j];
						hit_level = i;
						tlb_hit = true;
					}
				}
				else if (!instruction)
				{
					bool tlb_stores_data = !(tlbs[i][j]->getType() == TLBtype::Instruction);
					if (tlb_stores_data)
					{
						tlb_block_info = tlbs[i][j]->lookup(address, time, count, lock, eip, modeled, count, NULL);
						if (tlb_block_info != NULL)
						{
							tlb_block_info_hit = tlb_block_info;
							hit_tlb = tlbs[i][j];
							hit_level = i;
							tlb_hit = true;
						}
					}
				}
			}
			if (tlb_hit)
			{
				mmu_nest_log->debug("[MMU] TLB Hit at level: %d at TLB %s", hit_level, hit_tlb->getName().c_str());
				break;
			}
		}

		// ====================================================================
		// PHASE 2: Charge Hardware TLB Latency
		// ====================================================================

		SubsecondTime charged_tlb_latency = SubsecondTime::Zero();

		if (tlb_hit)
		{
			// ----- HW TLB HIT: Charge latency up to hit level -----
			page_size_result = tlb_block_info_hit->getPageSize();
			ppn_result       = tlb_block_info_hit->getPPN();

			if (instruction)
				tlbs = tlb_subsystem->getInstructionPath();
			else
				tlbs = tlb_subsystem->getDataPath();

			SubsecondTime tlb_latency[hit_level + 1];

			for (int i = 0; i < hit_level; i++)
			{
				for (UInt32 j = 0; j < tlbs[i].size(); j++)
				{
					tlb_latency[i] = max(tlbs[i][j]->getLatency(), tlb_latency[i]);
				}
				translation_stats.total_tlb_latency += tlb_latency[i];
				translation_stats.tlb_latency_per_level[i] += tlb_latency[i];
				charged_tlb_latency += tlb_latency[i];
			}

			for (UInt32 j = 0; j < tlbs[hit_level].size(); j++)
			{
				if (tlbs[hit_level][j] == hit_tlb)
				{
					translation_stats.total_tlb_latency += hit_tlb->getLatency();
					charged_tlb_latency += hit_tlb->getLatency();
					translation_stats.tlb_latency_per_level[hit_level] += hit_tlb->getLatency();
				}
			}

			// Advance the clock to reflect TLB-latency consumption
			shmem_perf_model->setElapsedTime(ShmemPerfModel::_USER_THREAD, time + charged_tlb_latency);
			if (count)
				translation_stats.total_translation_latency += charged_tlb_latency;

		}
		else
		{
			/*
			 * 2a) TLB Miss => sum the latency for all TLB levels. We do not break out at a 
			 *     partial level because we "visited" them all searching for a hit.
			 */
			SubsecondTime tlb_latency[tlbs.size()];

			for (UInt32 i = 0; i < tlbs.size(); i++)
			{
				for (UInt32 j = 0; j < tlbs[i].size(); j++)
				{
					tlb_latency[i] = max(tlbs[i][j]->getLatency(), tlb_latency[i]);
				}
				mmu_nest_log->debug("[MMU] Charging TLB Latency: %lu at level: %u", tlb_latency[i].getNS(), i);
				translation_stats.total_tlb_latency += tlb_latency[i];
				charged_tlb_latency += tlb_latency[i];
			}

			shmem_perf_model->setElapsedTime(ShmemPerfModel::_USER_THREAD, time + charged_tlb_latency);
			if (count)
				translation_stats.total_translation_latency += charged_tlb_latency;

			mmu_nest_log->debug("[MMU] New time after charging TLB latency: %lu", shmem_perf_model->getElapsedTime(ShmemPerfModel::_USER_THREAD).getNS());
		}

		/*
		 * 3) If TLB miss, we attempt the software TLB(s). The code loops over each possible
		 *    page size in 'm_pom_tlb' to see if we have a hit. We measure the access latencies
		 *    (software_tlb_latency[i]) and pick either the one that hits or the max if all miss.
		 */
		bool nest_hit = false;
		SubsecondTime charged_nest_latency = SubsecondTime::Zero();
		UInt16 asid = static_cast<UInt16>(core->getThread()->getAppId());

		if (!tlb_hit && m_nest->isEnabled())
		{
			SubsecondTime time_before_nest = shmem_perf_model->getElapsedTime(ShmemPerfModel::_USER_THREAD);
			
			if (count)
				translation_stats.nest_lookups++;

			nest_hit = m_nest->lookup(asid, address, ppn_result, page_size_result);

			if (count)
			{
				if (nest_hit) translation_stats.nest_hits++;
				else          translation_stats.nest_misses++;
			}

			charged_nest_latency = m_nest->getLatency();

			shmem_perf_model->setElapsedTime(ShmemPerfModel::_USER_THREAD,
				time_before_nest + charged_nest_latency);

			if (count)
			{
				translation_stats.software_tlb_latency += charged_nest_latency;
				translation_stats.total_translation_latency += charged_nest_latency;
			}
		}

		/*
		 * 4) If both hardware TLB and software TLB miss, we do a page table walk (PTW).
		 *    This includes possibly incurring a page fault if the memory is not resident,
		 *    and also a possible delay if all PT walkers are busy (pt_walkers->getSlotAllocationDelay).
		 */
		SubsecondTime total_fault_latency = SubsecondTime::Zero();
		SubsecondTime total_walk_latency  = SubsecondTime::Zero();
		bool translation_had_fault = false;

		if (!tlb_hit && !nest_hit)
		{
			SubsecondTime time_for_pt = shmem_perf_model->getElapsedTime(ShmemPerfModel::_USER_THREAD);

			MSHREntry pt_walker_entry; 
			pt_walker_entry.request_time = time_for_pt;

			// Possibly wait if all PT walkers are busy
			SubsecondTime delay = pt_walkers->getSlotAllocationDelay(pt_walker_entry.request_time);
			shmem_perf_model->setElapsedTime(ShmemPerfModel::_USER_THREAD, time_for_pt + delay);
			if(count)
				translation_stats.total_translation_latency += delay;

			// Perform the PT walk
			int app_id = core->getThread()->getAppId();
			PageTable* page_table = Sim()->getMimicOS()->getPageTable(app_id);
			bool userspace_mimicos_enabled = Sim()->getMimicOS()->isUserspaceMimicosEnabled();

			// @kanellok: restart_walk_after_fault is now false by default
			// The MMU handles page faults and restarts the walk itself
			const bool restart_walk_upon_page_fault = false;
			bool caused_page_fault = false;
			bool had_page_fault = false;  // Persists across loop iterations for fault latency tracking
			
			// Declare ptw_result outside the loop so it's accessible after
			PTWOutcome ptw_result;

			// Loop to handle page faults: perform PTW, handle fault if needed, retry
			do {
				ptw_result = performPTW(address, modeled, count, false, eip, lock, page_table, restart_walk_upon_page_fault, instruction);

			total_walk_latency = ptw_result.latency;
			if (count)
			{
				translation_stats.total_walk_latency += total_walk_latency;
				translation_stats.page_table_walks++;
			}

			caused_page_fault = ptw_result.page_fault;
			if (caused_page_fault)
			{
				had_page_fault = true;  // Track that a fault occurred
				mmu_nest_log->debug("[MMU] Page Fault occured");
				if (count)
					translation_stats.page_faults++;
				
				// Handle page fault at MMU level (sniper-space mode)
				if (!userspace_mimicos_enabled)
				{
					mmu_nest_log->debug("[MMU] Handling page fault in sniper-space mode, calling exception handler");
					ExceptionHandlerBase *handler = Sim()->getCoreManager()->getCoreFromID(core->getId())->getExceptionHandler();

					ExceptionHandlerBase::FaultCtx fault_ctx{};
					fault_ctx.vpn = address >> 12;
					fault_ctx.page_table = page_table;
					fault_ctx.alloc_in.metadata_frames = ptw_result.requested_frames;
					handler->handle_page_fault(fault_ctx);
					
					mmu_nest_log->debug("[MMU] Page fault handled, restarting PTW for address: %lx", address);
					// Loop will retry PTW after fault is handled
				}
			}
			} while (caused_page_fault && !userspace_mimicos_enabled);
			translation_had_fault = had_page_fault;
			if (had_page_fault && count)
				translation_stats.translations_with_fault++;
			
			// Calculate fault latency for timing (after loop, fault should be resolved)
			if (had_page_fault && !userspace_mimicos_enabled)
			{
				SubsecondTime m_page_fault_latency = Sim()->getMimicOS()->getPageFaultLatency();
				if (count)
				{
					translation_stats.total_fault_latency += m_page_fault_latency;
				}
				total_fault_latency = m_page_fault_latency;
			}

			pt_walker_entry.completion_time = time_for_pt + delay 
			                                  + total_walk_latency
			                                  + total_fault_latency;

			pt_walkers->allocate(pt_walker_entry);

			// Move the time to the end of the PTW + possible page fault
			if (had_page_fault && !userspace_mimicos_enabled)
			{
				PseudoInstruction *i = new PageFaultRoutineInstruction(total_fault_latency);
				getCore()->getPerformanceModel()->queuePseudoInstruction(i);
				shmem_perf_model->setElapsedTime(ShmemPerfModel::_USER_THREAD, pt_walker_entry.completion_time);
				if(count)
					translation_stats.total_translation_latency += total_walk_latency;
			}
			else
			{
				shmem_perf_model->setElapsedTime(ShmemPerfModel::_USER_THREAD, pt_walker_entry.completion_time);
				if(count)
					translation_stats.total_translation_latency += total_walk_latency;
			}

			ppn_result       = ptw_result.ppn;
			page_size_result = ptw_result.page_size;

		}

		// 5) Allocate the translation in intermediate TLB levels (if needed).
		if (instruction)
			tlbs = tlb_subsystem->getInstructionPath();
		else
			tlbs = tlb_subsystem->getDataPath();

		std::map<int, vector<EvictedTranslation>> evicted_translations;
		int tlb_levels = tlbs.size();

		// Some TLB hierarchies might have a "prefetch" TLB, skipping the last level for 
		// "allocate on miss." We handle that with isPrefetchEnabled checks if relevant.
		if (tlb_subsystem->isPrefetchEnabled())
		{
			tlb_levels = tlbs.size() - 1;
		}

		// For each TLB level, we insert or attempt to insert the final translation 
		// if "allocate on miss" is set and if it supports our page size.
		for (int i = 0; i < tlb_levels; i++)
		{
			for (UInt32 j = 0; j < tlbs[i].size(); j++)
			{
				// If there's any "evicted" translation from the previous level, we attempt to place it here
				if ((i > 0) && (!evicted_translations[i - 1].empty()))
				{
					TLBAllocResult result;

					mmu_nest_log->debug("[MMU] There are evicted translations from level: %d", (i - 1));
					for (const EvictedTranslation &evicted : evicted_translations[i - 1])
					{
						mmu_nest_log->debug("[MMU] Evicted Translation: %lx", evicted.address);
						IntPtr evicted_address = evicted.address;
						int evicted_page_size = evicted.page_size;
						IntPtr evicted_ppn = evicted.ppn;
						if (tlbs[i][j]->supportsPageSize(evicted_page_size))
						{
							mmu_nest_log->debug("[MMU] Allocating evicted entry in TLB: Level = %d Index =  %d", i, j);
							result = tlbs[i][j]->allocate(evicted_address, time, count, lock,
							                              evicted_page_size, evicted_ppn);

							if (result.evicted)
							{
								evicted_translations[i].push_back(
									EvictedTranslation(result.address, result.page_size, result.ppn));
							}
						}
					}
				}

				// If the TLB can store this page size, is "allocate on miss," and we either missed 
				// or want to replicate the translation in lower levels:
				if (tlbs[i][j]->supportsPageSize(page_size_result)
				    && tlbs[i][j]->getAllocateOnMiss()
				    && (!tlb_hit || (tlb_hit && hit_level > i)))
				{
					mmu_nest_log->debug("[MMU] %s supports page size: %d", tlbs[i][j]->getName().c_str(), page_size_result);
					mmu_nest_log->debug("[MMU] Allocating in TLB: Level = %d Index = %d with page size: %d and VPN: %lx", i, j, page_size_result, (address >> page_size_result));
					TLBAllocResult result = tlbs[i][j]->allocate(address, time, count, lock, page_size_result, ppn_result);
					if (result.evicted)
					{
						evicted_translations[i].push_back(
							EvictedTranslation(result.address, result.page_size, result.ppn));
					}
				}
			}
		}

		/*
		 * 6) Allocate the translation in the software TLB if we had a miss there. 
		 *    We check each software TLB if it supports page_size_result and is "allocate on miss."
		 */
		if (!nest_hit && !tlb_hit && m_nest->isEnabled())
		{
			size_t kicks = 0;
			bool evicted = false;
			m_nest->insert(asid, address, ppn_result, page_size_result, kicks, evicted);

			if (count)
			{
				translation_stats.nest_insertions++;
				translation_stats.nest_kicks += kicks;
				if (evicted)
					translation_stats.nest_evictions++;
			}
		}

		// ====================================================================
		// PHASE 6: Calculate Physical Address
		// ====================================================================

		// Physical address = PPN * base_page_size + offset
		// PPN is always at 4KB granularity
		// Offset is extracted based on actual page size (4KB or 2MB)

		const int page_size_in_bytes = 1 << page_size_result;      // 2^page_size (optimized from pow())
		constexpr int base_page_size_in_bytes = 1 << 12;           // 4KB base page

		IntPtr final_physical_address = (ppn_result * base_page_size_in_bytes)
		                                + (address % page_size_in_bytes);

		mmu_nest_log->debug("[MMU] Physical Address: %lx PPN: %lx Page Size: %d", final_physical_address, ppn_result*base_page_size_in_bytes, page_size_result);
		mmu_nest_log->debug("[MMU] Total translation latency: %lu", (charged_tlb_latency + charged_nest_latency + total_walk_latency).getNS());
		mmu_nest_log->debug("[MMU] ---- Ending address translation for virtual address: %lx ----", address);

		if (m_trace_file && !tlb_hit && count)
		{
			NestTraceRecord rec;
			rec.time_ns   = shmem_perf_model->getElapsedTime(ShmemPerfModel::_USER_THREAD).getNS();
			rec.vpn4k     = address >> 12;
			rec.asid      = asid;
			rec.page_size = static_cast<UInt8>(page_size_result);
			rec.flags     = (nest_hit ? 0x1 : 0x0) | (translation_had_fault ? 0x2 : 0x0);
			fwrite(&rec, sizeof(rec), 1, m_trace_file);
		}

		return final_physical_address;
	}

// ============================================================================
// Page Table Walk Support
// ============================================================================

    /**
     * @brief Filter PTW result through Page Walk Cache (PWC).
     *
     * Removes intermediate page table entries that are already cached in the
     * PWC, reducing the effective walk latency.
     *
     * @param address     Virtual address being translated
     * @param ptw_result  Raw PTW result from page table
     * @param page_table  Page table being walked
     * @param count       Whether to count statistics
     * @return Filtered PTW result with cached entries removed
     */
	PTWResult MemoryManagementUnitNEST::filterPTWResult(IntPtr address, PTWResult ptw_result,
	                                                      PageTable *page_table,
	                                                      bool count)
	{
		return ptw_filter->filterPTWResult(address, ptw_result, page_table, count);
	}

    /**
     * @brief Get the page table for the current thread.
     *
     * @return Pointer to the current thread's page table
     */
	PageTable *MemoryManagementUnitNEST::getPageTable()
	{
		int app_id = core->getThread()->getAppId();
		return Sim()->getMimicOS()->getPageTable(app_id);
	}

    /**
     * @brief Discover Virtual Memory Areas (VMAs).
     *
     * Not implemented for this MMU variant. VMAs are used by range-based
     * MMUs for coalescing translations.
     */
	void MemoryManagementUnitNEST::discoverVMAs()
	{
	}

} // namespace ParametricDramDirectoryMSI
