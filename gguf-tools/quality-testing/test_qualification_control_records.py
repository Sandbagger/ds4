#!/usr/bin/env python3
"""Bounded raw-wire observations for the qualification control parent."""

from __future__ import annotations

import array
import hashlib
import os
import socket
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_compact_runtime_qualify as existing


TOOL = existing.TOOL
QUALIFICATION_CONTROL_MESSAGE = existing.QUALIFICATION_CONTROL_MESSAGE
_wire_message = existing._qualification_control_wire_message


class QualificationControlWireRecordsTest(unittest.TestCase):
    """Exercise raw observations without a child process or invocation thread."""

    def _helper(self) -> existing.QualificationControlParentContractTest:
        return existing.QualificationControlParentContractTest()

    def _send_rights(
        self, child: socket.socket, payload: bytes, descriptor: int
    ) -> None:
        sent = child.sendmsg(
            [payload],
            [
                (
                    socket.SOL_SOCKET,
                    socket.SCM_RIGHTS,
                    array.array("i", (descriptor,)),
                )
            ],
        )
        self.assertEqual(sent, len(payload))

    def _recv_exact(self, child: socket.socket, size: int) -> bytes:
        payload = bytearray()
        while len(payload) < size:
            part = child.recv(size - len(payload))
            self.assertTrue(part, "qualification control peer disconnected")
            payload.extend(part)
        return bytes(payload)

    def _assert_received_descriptors_closed(
        self, closed: list[int], sender_descriptor: int
    ) -> None:
        self.assertTrue(closed, "received SCM_RIGHTS descriptor was not closed")
        self.assertTrue(
            any(descriptor != sender_descriptor for descriptor in closed),
            "the observation must close the received descriptor, not the sender fd",
        )
        for descriptor in set(closed):
            if descriptor == sender_descriptor:
                continue
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def _capture_os_closes(self) -> tuple[list[int], object]:
        closed: list[int] = []
        real_close = TOOL.os.close

        def close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        patcher = mock.patch.object(TOOL.os, "close", side_effect=close)
        patcher.start()
        return closed, patcher

    def _close_channel(self, control: object, child: socket.socket) -> None:
        control.close()  # type: ignore[union-attr]
        child.close()

    def test_clock_interrupts_propagate_in_deadline_and_receive_wait(self) -> None:
        for fail_on_call in (1, 2):
            with self.subTest(clock_call=fail_on_call):
                calls = 0
                interrupted = KeyboardInterrupt("stop control clock")

                def clock() -> float:
                    nonlocal calls
                    calls += 1
                    if calls == fail_on_call:
                        raise interrupted
                    return 0.0

                with TOOL.QualificationControl.create(
                    timeout_seconds=0.1, monotonic=clock,
                ) as control:
                    with self.assertRaises(KeyboardInterrupt) as raised:
                        control.receive_model()
                    self.assertIs(raised.exception, interrupted)

    def test_model_fd_and_ack_records_preserve_exact_wire_bytes_and_counts(
        self,
    ) -> None:
        self.assertEqual(TOOL.MAX_QUALIFICATION_CONTROL_WIRE_BYTES, 65536)
        helper = self._helper()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            payload = b"wire-record model\0" * 32
            model.write_bytes(payload)
            descriptor = os.open(model, os.O_RDONLY)
            control, child = helper._channel_and_child(timeout_seconds=1.0)
            try:
                identity = helper._send_model(child, descriptor)
                evidence = control.receive_model()
                model_wire = _wire_message(1, 0, identity)
                ack_wire = _wire_message(6, 0, identity)
                self.assertEqual(
                    self._recv_exact(child, QUALIFICATION_CONTROL_MESSAGE.size),
                    ack_wire,
                )
                self.assertEqual(
                    control.wire_records,
                    (
                        {
                            "direction": "receive",
                            "payload": model_wire,
                            "complete": True,
                            "descriptor_count": 1,
                        },
                        {
                            "direction": "send",
                            "payload": ack_wire,
                            "complete": True,
                            "descriptor_count": 0,
                        },
                    ),
                )
                self.assertEqual(evidence.sha256, hashlib.sha256(payload).hexdigest())
            finally:
                self._close_channel(control, child)
                os.close(descriptor)

    def test_preloaded_ready_result_records_follow_corresponding_ack_order(self) -> None:
        helper = self._helper()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"preloaded checkpoint model")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = helper._channel_and_child(timeout_seconds=1.0)
            try:
                identity = helper._send_model(child, descriptor)
                evidence = control.receive_model()
                model_wire = _wire_message(1, 0, identity)
                model_ack = _wire_message(6, 0, identity)
                self.assertEqual(
                    self._recv_exact(child, QUALIFICATION_CONTROL_MESSAGE.size),
                    model_ack,
                )
                ready_wire = _wire_message(2, 17)
                result_wire = _wire_message(4, 17, evidence.identity)
                child.sendall(ready_wire + result_wire)

                before, after = control.bracket_sample(
                    17,
                    capture_before=lambda: "before",
                    capture_after=lambda: "after",
                )
                ready_ack = _wire_message(3, 17)
                result_ack = _wire_message(5, 17)
                self.assertEqual(
                    self._recv_exact(child, QUALIFICATION_CONTROL_MESSAGE.size),
                    ready_ack,
                )
                self.assertEqual(
                    self._recv_exact(child, QUALIFICATION_CONTROL_MESSAGE.size),
                    result_ack,
                )
                self.assertEqual((before, after), ("before", "after"))
                self.assertEqual(
                    control.wire_records,
                    (
                        {"direction": "receive", "payload": model_wire, "complete": True, "descriptor_count": 1},
                        {"direction": "send", "payload": model_ack, "complete": True, "descriptor_count": 0},
                        {"direction": "receive", "payload": ready_wire, "complete": True, "descriptor_count": 0},
                        {"direction": "send", "payload": ready_ack, "complete": True, "descriptor_count": 0},
                        {"direction": "receive", "payload": result_wire, "complete": True, "descriptor_count": 0},
                        {"direction": "send", "payload": result_ack, "complete": True, "descriptor_count": 0},
                    ),
                )
            finally:
                self._close_channel(control, child)
                os.close(descriptor)

    def test_malformed_full_and_eof_partial_prefixes_are_bounded_and_close_rights(
        self,
    ) -> None:
        helper = self._helper()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"malformed and partial model")
            descriptor = os.open(model, os.O_RDONLY)
            try:
                identity = os.fstat(descriptor)
                malformed = bytearray(_wire_message(1, 0, identity))
                malformed[8:12] = struct.pack("@I", QUALIFICATION_CONTROL_MESSAGE.size + 1)
                control, child = helper._channel_and_child(timeout_seconds=1.0)
                closed, patcher = self._capture_os_closes()
                try:
                    self._send_rights(child, bytes(malformed), descriptor)
                    with self.assertRaises(ValueError):
                        control.receive_model()
                    self.assertEqual(
                        control.wire_records,
                        (
                            {
                                "direction": "receive",
                                "payload": bytes(malformed),
                                "complete": True,
                                "descriptor_count": 1,
                            },
                        ),
                    )
                    self._assert_received_descriptors_closed(closed, descriptor)
                    os.fstat(descriptor)
                finally:
                    patcher.stop()
                    self._close_channel(control, child)

                partial = _wire_message(1, 0, identity)[:13]
                control, child = helper._channel_and_child(timeout_seconds=1.0)
                closed, patcher = self._capture_os_closes()
                try:
                    self._send_rights(child, partial, descriptor)
                    child.close()
                    with self.assertRaises(ValueError):
                        control.receive_model()
                    self.assertEqual(
                        control.wire_records,
                        (
                            {
                                "direction": "receive",
                                "payload": partial,
                                "complete": False,
                                "descriptor_count": 1,
                            },
                        ),
                    )
                    self._assert_received_descriptors_closed(closed, descriptor)
                    os.fstat(descriptor)
                finally:
                    patcher.stop()
                    control.close()
            finally:
                os.close(descriptor)

    def test_records_are_deep_copied_and_stable_after_close(self) -> None:
        helper = self._helper()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"stable wire records")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = helper._channel_and_child(timeout_seconds=1.0)
            try:
                identity = helper._send_model(child, descriptor)
                control.receive_model()
                self._recv_exact(child, QUALIFICATION_CONTROL_MESSAGE.size)
                expected = control.wire_records
                exposed = control.wire_records
                self.assertIsInstance(exposed, tuple)
                self.assertIsNot(exposed, expected)
                self.assertIsNot(exposed[0], expected[0])
                exposed[0]["direction"] = "send"
                exposed[0]["payload"] = b"mutated"
                exposed[0]["complete"] = False
                exposed[0]["descriptor_count"] = 99
                self.assertEqual(control.wire_records, expected)

                control.close()
                after_close = control.wire_records
                self.assertEqual(after_close, expected)
                after_close[0]["payload"] = b"changed after close"
                self.assertEqual(control.wire_records, expected)
                self.assertEqual(expected[0]["payload"], _wire_message(1, 0, identity))
            finally:
                self._close_channel(control, child)
                os.close(descriptor)

    def test_wire_budget_refusal_closes_rights_before_any_ack_and_is_sticky(self) -> None:
        helper = self._helper()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"budget model")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = helper._channel_and_child(timeout_seconds=1.0)
            closed, patcher = self._capture_os_closes()
            try:
                identity = helper._send_model(child, descriptor)
                model_wire = _wire_message(1, 0, identity)
                ack_wire = _wire_message(6, 0, identity)
                with mock.patch.object(
                    TOOL,
                    "MAX_QUALIFICATION_CONTROL_WIRE_BYTES",
                    len(model_wire) + len(ack_wire) - 1,
                ):
                    with self.assertRaisesRegex(ValueError, "wire|budget|cap|limit"):
                        control.receive_model()
                self.assertEqual(child.recv(1), b"", "budget refusal sent a false ACK")
                self.assertEqual(
                    control.wire_records,
                    (
                        {
                            "direction": "receive",
                            "payload": model_wire,
                            "complete": True,
                            "descriptor_count": 1,
                        },
                    ),
                )
                self.assertLessEqual(
                    sum(len(record["payload"]) for record in control.wire_records),
                    len(model_wire) + len(ack_wire) - 1,
                )
                self._assert_received_descriptors_closed(closed, descriptor)
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    control.receive_model()
            finally:
                patcher.stop()
                self._close_channel(control, child)
                os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
