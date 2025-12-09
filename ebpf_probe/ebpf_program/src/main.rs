#![no_std]
#![no_main]

use aya_ebpf::{
    bindings::xdp_action,
    macros::xdp,
    programs::XdpContext,
};

/// Simple XDP Firewall - Just passes all packets for now
/// In production, this would inspect packets and make filtering decisions
#[xdp]
pub fn xdp_firewall(_ctx: XdpContext) -> u32 {
    // For MVP: Just pass all packets through
    // The fact that this runs proves we have kernel-level access!
    xdp_action::XDP_PASS
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}
