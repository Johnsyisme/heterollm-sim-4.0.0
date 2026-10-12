"""Opt-in finite NAND write-buffer semantics."""

import unittest

from heterollm_sim.memory_types import AccessRequest, Operation, make_hbf_config
from heterollm_sim.nand_core import NandCore


PAGE = 16 * 1024


def _config(**kwargs):
    values = dict(
        channels=1,
        targets_per_channel=1,
        dies_per_target=1,
        luns_per_die=1,
        planes_per_lun=1,
        blocks_per_plane=1,
        pages_per_block=8,
        page_bytes=PAGE,
        host_granularity_bytes=4 * 1024,
        host_bandwidth_gb_s=10.0,
        internal_bandwidth_gb_s=10.0,
        page_read_ns=100.0,
        page_program_ns=200.0,
        write_buffer_bytes=PAGE,
        metadata={"buffered_write_mode": True},
    )
    values.update(kwargs)
    return make_hbf_config(**values)


class WriteBufferSemanticsTests(unittest.TestCase):
    def test_small_writes_merge_and_reads_hit_staging_until_drain(self):
        core = NandCore(_config())
        first = core.execute(AccessRequest("w0", Operation.WRITE, 0, 4096))
        second = core.execute(AccessRequest("w1", Operation.WRITE, 4096, 4096))

        self.assertEqual(first.counters["write_completion"], "host_buffer_ack_pending_media_drain")
        self.assertEqual(first.physical_programs, 0)
        self.assertEqual(second.merge_bytes, 4096)
        self.assertEqual(core.buffered_page_count, 1)
        self.assertEqual(core.pending_write_buffer_bytes, 8192)

        staged_read = core.execute(AccessRequest("r0", Operation.READ, 0, PAGE))
        self.assertEqual(staged_read.staging_read_hits, 1)
        self.assertEqual(staged_read.physical_read_bytes, 0)
        self.assertNotIn("PAGE_READ", {stage.name for stage in staged_read.stages})

        drained = core.flush_write_buffer()
        self.assertEqual(drained["flushed_pages"], 1)
        self.assertEqual(drained["merge_bytes"], 4096)
        self.assertEqual(drained["physical_programs"], 1)
        self.assertEqual(drained["physical_write_bytes"], PAGE)
        self.assertGreater(drained["media_drain_completion_ns"], second.host_ack_completion_ns)
        self.assertEqual(core.pending_write_buffer_bytes, 0)

        media_read = core.execute(AccessRequest("r1", Operation.READ, 0, PAGE))
        self.assertEqual(media_read.staging_read_hits, 0)
        self.assertEqual(media_read.physical_read_bytes, PAGE)

    def test_full_buffer_auto_drains_before_next_host_ack(self):
        core = NandCore(_config(write_buffer_bytes=8192))
        core.execute(AccessRequest("w0", Operation.WRITE, 0, 4096))
        core.execute(AccessRequest("w1", Operation.WRITE, 4096, 4096))
        next_write = core.execute(AccessRequest("w2", Operation.WRITE, 8192, 4096))

        self.assertGreater(next_write.write_buffer_wait_ns, 0.0)
        self.assertEqual(next_write.counters["write_completion"], "host_buffer_ack_pending_media_drain")
        self.assertLessEqual(next_write.write_buffer_occupancy_bytes, 8192)

    def test_switch_is_opt_in_and_default_remains_write_through(self):
        core = NandCore(_config(metadata={}, write_buffer_bytes=PAGE))
        result = core.execute(AccessRequest("w", Operation.WRITE, 0, 4096))

        self.assertEqual(result.counters["write_completion"], "media_program_complete")
        self.assertEqual(result.physical_programs, 1)
        self.assertEqual(core.buffered_page_count, 0)
        self.assertEqual(core.flush_write_buffer()["buffered_write_mode"], False)


if __name__ == "__main__":
    unittest.main()
