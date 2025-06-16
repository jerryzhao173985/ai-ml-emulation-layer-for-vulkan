# ML Emulation Layer for Vulkan®

Arm® has approached the Khronos® group with a set of Machine Learning
extensions for the Vulkan® and SPIR-V™ APIs. On devices where these extensions
have not been implemented by the Vulkan® Installable Device Drivers (ICD), the
ML Emulation Layer is required.

The ML Emulation Layer for Vulkan® provides an implementation of the ML APIs
enabling ML workloads to be executed on any Vulkan® Compute capable device. The
Emulation Layer is split into separate graph and tensor layers that are inserted
by the Vulkan® Loader.

## Building

Please see [Build](BUILD.md).

## License

The ML Emulation Layer for Vulkan® is provided under an Apache-2.0 license.
Please see [Apache-2.0.txt](LICENSES/Apache-2.0.txt) for more information.

## Contributions

Please see [Contributing](CONTRIBUTING.md).

## Security

Please see [Security](SECURITY.md).

## Trademark notice

Arm® is a registered trademarks of Arm Limited (or its subsidiaries) in the US
and/or elsewhere.

Khronos®, Vulkan® and SPIR-V™ are registered trademarks of the
[Khronos® Group](https://www.khronos.org/legal/trademarks).
