
#pragma once
#include "memory_manager.h"
#include "cache_cntlr.h"
#include "subsecond_time.h"
#include "fixed_types.h"
#include "core.h"
#include "shmem_perf_model.h"
#include "pagetable.h"
#include "tlb_subsystem.h"
#include "mmu_base.h"
#include "ptmshrs.h"
#include "base_filter.h"
#include "sim_log.h"
#include "debug_config.h"

#include <vector>
#include <cstdint>
#include <cstddef>

namespace ParametricDramDirectoryMSI
{
	class TLBHierarchy;

	class NestCuckooTable
	{
		public:
			static constexpr size_t NUM_BANKS = 4;
			struct Entry
			{
				bool valid;
				IntPtr vpn;
				IntPtr ppn;

				Entry() : valid(false), vpn(0), ppn(0) {}
			};

			struct Stats
			{
				UInt64 lookups;
				UInt64 hits;
				UInt64 misses;
				UInt64 insertions;
				UInt64 evictions;
			};
		private:
			int m_page_size;
			size_t m_total_entries;
			size_t m_entries_per_bank;
			std::vector<Entry> m_banks[NUM_BANKS];
			uint8_t m_evict_counter;
			Stats m_stats;

			size_t hash0(IntPtr vpn) const
			{
				uint64_t x = static_cast<uint64_t>(vpn);
				x ^= (x >> 16);
				return static_cast<size_t>(x % m_entries_per_bank);
			}

			size_t hash1(IntPtr vpn) const
			{
				uint64_t x = static_cast<uint64_t>(vpn) * 0x9e3779b97f4a7c15ULL;
				x ^= (x >> 21);
				return static_cast<size_t>(x % m_entries_per_bank);
			}

			size_t hash2(IntPtr vpn) const
			{
				uint64_t x = static_cast<uint64_t>(vpn) * 0xbf58476d1ce4e5b9ULL;
				x ^= (x >> 27);
				return static_cast<size_t>(x % m_entries_per_bank);
			}

			size_t hash3(IntPtr vpn) const
			{
				uint64_t x = static_cast<uint64_t>(vpn);
				x = (x << 13) | (x >> 51);
				x ^= 0x94d049bb133111ebULL;
				return static_cast<size_t>(x % m_entries_per_bank);
			}

			size_t getIndex(size_t bank, IntPtr vpn) const
			{
				switch (bank)
				{
					case 0: return hash0(vpn);
					case 1: return hash1(vpn);
					case 2: return hash2(vpn);
					case 3: return hash3(vpn);
					default: return 0;
				}
			}
		
		public:
			NestCuckooTable(int page_size, size_t total_entries)
			: m_page_size(page_size),
			  m_total_entries(total_entries),
			  m_entries_per_bank(total_entries / NUM_BANKS),
			  m_evict_counter(0)
			{
				for (size_t b = 0; b < NUM_BANKS; b++)
					m_banks[b].resize(m_entries_per_bank);
				resetStats();
			}

			int getPageSize() const { return m_page_size; }

			bool lookup(IntPtr vpn, IntPtr &out_ppn)
			{
				m_stats.lookups++;
				for (size_t b = 0; b < NUM_BANKS; b++)
				{
					size_t idx = getIndex(b, vpn);
					const Entry &e = m_banks[b][idx];
					if (e.valid && e.vpn == vpn)
					{
						out_ppn = e.ppn;
						m_stats.hits++;
						return true;
					}
				}
				m_stats.misses++;
				return false;
			}

			void insert(IntPtr vpn, IntPtr ppn)
			{

				m_stats.insertions++;

				for (size_t b = 0; b < NUM_BANKS; b++)
				{
					size_t idx = getIndex(b, vpn);
					Entry &e = m_banks[b][idx];
					if (e.valid && e.vpn == vpn)
					{
						e.ppn = ppn;
						return;
					}
				}

				for (size_t b = 0; b < NUM_BANKS; b++)
				{
					size_t idx = getIndex(b, vpn);
					Entry &e = m_banks[b][idx];
					if (!e.valid)
					{
						e.valid = true;
						e.vpn = vpn;
						e.ppn = ppn;
						return;
					}
				}

				size_t evict_bank = m_evict_counter % NUM_BANKS;
				m_evict_counter++;
				size_t evict_idx = getIndex(evict_bank, vpn);

				Entry &victim = m_banks[evict_bank][evict_idx];
				victim.valid = true;
				victim.vpn = vpn;
				victim.ppn = ppn;
				m_stats.evictions++;
			}
			void resetStats() { m_stats = Stats{0, 0, 0, 0, 0}; }
    		const Stats& getStats() const { return m_stats; }
	};

	class MemoryManagementUnitNEST : public MemoryManagementUnitBase
	{

	private:
		MemoryManagerBase *memory_manager;
		TLBHierarchy *tlb_subsystem;

		MSHR *pt_walkers; 
		BaseFilter *ptw_filter;

		NestCuckooTable **m_nest_tables;
		int m_num_page_sizes;


		TLB **m_pom_tlb;
		int pom_tlb_entry_size;
		IntPtr *software_tlb_base_register;
		UInt32 *m_size;
		UInt32 *m_associativity;

		SimLog *mmu_pomtlb_log;

		struct
		{
			UInt64 num_translations;

			UInt64 page_faults;
			UInt64 page_table_walks;
			
			SubsecondTime total_walk_latency;
			SubsecondTime total_walk_delay_latency;
			SubsecondTime total_fault_latency;
			SubsecondTime total_translation_latency;
			SubsecondTime total_tlb_latency;
			SubsecondTime software_tlb_latency;
			
			SubsecondTime walker_is_active;
			SubsecondTime *tlb_latency_per_level;
			UInt64 *tlb_hit_page_sizes; 

		} translation_stats;

	public:
		MemoryManagementUnitNEST(Core *core, MemoryManagerBase *memory_manager, ShmemPerfModel *shmem_perf_model, String name, MemoryManagementUnitBase *nested_mmu);
		~MemoryManagementUnitNEST();

		void discoverVMAs();
		void instantiatePageTableWalker();
		void instantiateMetadataTable();
		void instantiateTLBSubsystem();
		void registerMMUStats();

		PTWResult filterPTWResult(IntPtr address, PTWResult ptw_result, PageTable *page_table, bool count);
		IntPtr performAddressTranslation(IntPtr eip, IntPtr address, bool instruction, Core::lock_signal_t lock, bool modeled, bool count);
		PageTable *getPageTable();
	};
}