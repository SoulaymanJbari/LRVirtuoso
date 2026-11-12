
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
#include <mutex>
#include <unordered_map>
#include <utility>
#include <cstdio>

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
				UInt16 asid;
				IntPtr vpn;
				IntPtr ppn;

				Entry() : valid(false), asid(0), vpn(0), ppn(0) {}
			};

			struct Stats
			{
				UInt64 lookups;
				UInt64 hits;
				UInt64 misses;
				UInt64 insertions;
				UInt64 evictions;
				UInt64 kicks;
			};
		private:
			int m_page_size;
			size_t m_total_entries;
			size_t m_entries_per_bank;
			size_t m_max_kicks;
			std::vector<Entry> m_banks[NUM_BANKS];
			uint8_t m_evict_counter;
			Stats m_stats;

			static inline uint64_t mix64(uint64_t z)
			{
				z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
				z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
				return z ^ (z >> 31);
			}

			static inline uint64_t makeKey(UInt16 asid, IntPtr vpn)
			{
				return static_cast<uint64_t>(vpn) ^ (static_cast<uint64_t>(asid) << 48);
			}

			size_t getIndex(size_t bank, uint64_t key) const
			{
				static const uint64_t seeds[NUM_BANKS] = {
					0x517cc1b727220a95ULL, 0x9e3779b97f4a7c15ULL,
					0xbf58476d1ce4e5b9ULL, 0x3141592653589793ULL };
				return static_cast<size_t>(mix64(key ^ seeds[bank]) % m_entries_per_bank);
			}
		
		public:
			NestCuckooTable(int page_size, size_t total_entries, size_t max_kicks)
			: m_page_size(page_size),
			  m_total_entries(total_entries),
			  m_entries_per_bank(total_entries / NUM_BANKS),
			  m_max_kicks(max_kicks),
			  m_evict_counter(0)
			{
				for (size_t b = 0; b < NUM_BANKS; b++)
					m_banks[b].resize(m_entries_per_bank);
				resetStats();
			}

			int getPageSize() const { return m_page_size; }
    		const Stats& getStats() const { return m_stats; }
			void resetStats() { m_stats = Stats{0, 0, 0, 0, 0, 0}; }

			bool lookup(UInt16 asid, IntPtr vpn, IntPtr &out_ppn)
			{
				m_stats.lookups++;
				uint64_t key = makeKey(asid, vpn);
				for (size_t b = 0; b < NUM_BANKS; b++)
				{
					const Entry &e = m_banks[b][getIndex(b, key)];
					if (e.valid && e.asid == asid && e.vpn == vpn)
					{
						out_ppn = e.ppn;
						m_stats.hits++;
						return true;
					}
				}
				m_stats.misses++;
				return false;
			}

			bool insert(UInt16 asid, IntPtr vpn, IntPtr ppn, size_t *kicks_out = nullptr)
			{

				m_stats.insertions++;
				if (kicks_out)
					*kicks_out = 0;

				uint64_t key = makeKey(asid, vpn);

				for (size_t b = 0; b < NUM_BANKS; b++)
				{
					Entry &e = m_banks[b][getIndex(b, key)];
					if (e.valid && e.asid == asid && e.vpn == vpn)
					{
						e.ppn = ppn;
						return false;
					}
				}

				for (size_t b = 0; b < NUM_BANKS; b++)
				{
					Entry &e = m_banks[b][getIndex(b, key)];
					if (!e.valid)
					{
						e.valid = true;
						e.asid = asid;
						e.vpn = vpn;
						e.ppn = ppn;
						return false;
					}
				}

				UInt16 curr_asid = asid;
				IntPtr curr_vpn = vpn;
				IntPtr curr_ppn = ppn;
				size_t curr_bank = m_evict_counter % NUM_BANKS;
				m_evict_counter++;

				size_t total_kicks = 0;
				for (size_t kick = 0; kick < m_max_kicks; kick++)
				{
					total_kicks++;
					m_stats.kicks++;

					Entry &slot = m_banks[curr_bank][getIndex(curr_bank, makeKey(curr_asid, curr_vpn))];
					std::swap(slot.asid, curr_asid);
					std::swap(slot.vpn,  curr_vpn);
					std::swap(slot.ppn,  curr_ppn);
					slot.valid = true;

					uint64_t curr_key = makeKey(curr_asid, curr_vpn);
					for (size_t i = 1; i < NUM_BANKS; i++)
					{
						size_t alt_bank = (curr_bank + i) % NUM_BANKS;
						Entry &alt = m_banks[alt_bank][getIndex(alt_bank, curr_key)];
						if (!alt.valid)
						{
							alt.valid = true;
							alt.asid = curr_asid;
							alt.vpn = curr_vpn;
							alt.ppn = curr_ppn;
							if (kicks_out)
								*kicks_out = total_kicks;
							return false;
						}
					}

					curr_bank = (curr_bank + 1) % NUM_BANKS;
				}

				m_stats.evictions++;
				if (kicks_out)
					*kicks_out = total_kicks;
				return true;
			}
	};

	class NestShared
	{
		public:
			enum class Mode { Off, Cuckoo, Ideal};

			static NestShared* get();

			Mode getMode() const {return m_mode; }
			bool isEnabled() const { return m_mode != Mode::Off;}
			SubsecondTime getLatency() const { return m_latency; }

			bool lookup(UInt16 asid, IntPtr address, IntPtr &ppn, int &page_size);
			void insert(UInt16 asid, IntPtr address, IntPtr ppn, int page_size,
						size_t &kicks, bool &evicted);
		
		private:
			NestShared();
			
			static uint64_t idealKey(UInt16 asid, IntPtr vpn)
			{
				return static_cast<uint64_t>(vpn) | (static_cast<uint64_t>(asid) << 48);
			}

			static NestShared *s_instance;
			static std::mutex s_init_lock;

			Mode m_mode;
			SubsecondTime m_latency;
			int m_num_page_sizes;
			std::vector<int> m_page_sizes;
			std::vector<NestCuckooTable*> m_tables;                       // mode Cuckoo
			std::vector<std::unordered_map<uint64_t, IntPtr>> m_ideal;    // mode Ideal
			std::mutex m_lock;
	};

	struct NestTraceRecord
	{
		UInt64 time_ns;
		UInt64 vpn4k;      // address >> 12
		UInt16 asid;
		UInt8  page_size;  // 12 ou 21
		UInt8  flags;      // bit0 : hit NEST, bit1 : défaut de page
	};

	class MemoryManagementUnitNEST : public MemoryManagementUnitBase
	{

	private:
		MemoryManagerBase *memory_manager;
		TLBHierarchy *tlb_subsystem;

		MSHR *pt_walkers; 
		BaseFilter *ptw_filter;

		NestShared *m_nest;
		FILE *m_trace_file;

		SimLog *mmu_nest_log;

		struct
		{
			UInt64 num_translations;
			UInt64 page_faults;
			UInt64 page_table_walks;

			UInt64 nest_lookups;
			UInt64 nest_hits;
			UInt64 nest_misses;
			UInt64 nest_insertions;
			UInt64 nest_kicks;
			UInt64 nest_evictions;
			
			SubsecondTime total_walk_latency;
			SubsecondTime total_walk_delay_latency;
			SubsecondTime total_fault_latency;
			SubsecondTime total_translation_latency;
			SubsecondTime total_tlb_latency;
			SubsecondTime software_tlb_latency;
			
			SubsecondTime walker_is_active;
			SubsecondTime *tlb_latency_per_level;
			UInt64 *tlb_hit_page_sizes; 

			UInt64 translations_with_fault;

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