# Walk-on animation proof

Place a new trigger test device and separate walk-on pads. Read existing placed
actors that might automatically animate the same player. Disable only the
competing behavior within the requested test, and save that instance change.
Changing a Verse class default does not override a serialized instance value.

For an ordinary Verse `trigger_device` editable, the observed editor layout is:

```
placed_actor.script_device_0.__verse_0xHASH_FieldName.savedActor
```

Discover the property wrapper through `DeviceToolset.GetDeviceProperties`.
Resolve that returned object, then read its `savedActor`. The returned wrapper
path is not the trigger assignment. Passing a `BP_Creative_Trigger_C` actor
directly to `SetDeviceProperty` fails its `trigger_device` type check.

The measured successful editor-Python operation was:

```python
with unreal.ScopedEditorTransaction('Wire animation test trigger'):
    placed_actor.modify()
    wrapper.modify()
    wrapper.set_editor_property('savedActor', trigger_actor)
assert wrapper.get_editor_property('savedActor') == trigger_actor
```

Validate the actual wrapper class and target actor before writing. Preserve an
existing different assignment unless replacing it is requested. Save the level;
read the assignment again after reopening when a reopen already occurs. The
equivalent generic ObjectTools operation sets the wrapper's `savedActor`
property with an actor `refPath`. Scalars can use `SetDeviceProperty` directly.

Build the final source, push changed level content, and follow cook/session
results. A push request being accepted does not establish successful cook.
The test should log trigger identity, actual configured clip identity, player
controller acquisition and start/completion/error. Use a per-player running
guard to keep a second pad from issuing a concurrent animation command.

On each intended surface, pair saved mesh/Skeleton/clip bindings with runtime
output and visible shoulder/elbow/wrist motion. Include floor contact and fingers
when claiming those. An authored log label is not a query of the engine's current
Skeleton object, and source FBX hashes do not identify a cooked actor by themselves.
