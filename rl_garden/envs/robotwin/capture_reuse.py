"""Reuse readbacks only within one LiveState.capture; never cache across exposures."""
class Camera:
    def __init__(self, camera, verify=False):
        self.camera, self.cache, self.verify = camera, {}, verify
        self.hits = 0

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def take_picture(self, *args, **kwargs):
        self.cache.clear()
        return self.camera.take_picture(*args, **kwargs)

    def get_picture(self, name):
        if name not in self.cache:
            self.cache[name] = self.camera.get_picture(name).copy()
        else:
            self.hits += 1
            if self.verify:
                import numpy as np
                assert np.array_equal(self.cache[name], self.camera.get_picture(name)), name
        # Preserve independent writable buffers for callers that modify their result.
        return self.cache[name].copy()


def install(verify=False):
    from rl_garden.envs.robotwin import laptop_estimated_state as live_state
    original = live_state.LiveState.capture
    assert not getattr(original, '_readback_reuse', False)
    def capture(reader):
        if reader.last_tick == reader.tick:
            return original(reader)
        cameras = reader.task().cameras
        names = ('static_camera_list', 'left_camera', 'right_camera')
        before = [getattr(cameras, name) for name in names]
        proxies = [Camera(c, verify) for c in before[0]]
        left, right = Camera(before[1], verify), Camera(before[2], verify)
        cameras.static_camera_list, cameras.left_camera, cameras.right_camera = proxies, left, right
        try:
            result = original(reader)
            hits = sum(c.hits for c in proxies + [left, right])
            reader.readback_reuse_hits = getattr(reader, 'readback_reuse_hits', 0) + hits
            if verify:
                assert hits > 0, 'No duplicate readbacks: optimization not exercised'
            return result
        finally:
            for name, value in zip(names, before):
                setattr(cameras, name, value)
    capture._readback_reuse = True
    live_state.LiveState.capture = capture
    def restore():
        if live_state.LiveState.capture is capture:
            live_state.LiveState.capture = original
    return restore
