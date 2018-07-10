//! Type-level safe mutable global access.
//!
//! ```
//! use global::Global;
//!
//! // The global value.
//! static VALUE: Global<i32> = Global::INIT;
//!
//! // Spawn 100 threads and join them all.
//! let mut threads = Vec::new();
//!
//! for _ in 0..100 {
//!     threads.push(std::thread::spawn(|| {
//!         *VALUE.lock() += 1;
//!     }));
//! }
//!
//! for thread in threads {
//!     thread.join().unwrap();
//! }
//!
//! // This value is guaranteed to be 100.
//! assert_eq!(*VALUE.lock(), 100);
//! ```

extern crate once_nonstatic;

use std::{
    sync::{Arc, Mutex, MutexGuard},
    ops::{Deref, DerefMut},
    cell::UnsafeCell,
    mem::ManuallyDrop,
};

use once_nonstatic::Once;

/// A global value.
///
/// All types wrapped in `Global` must implement [`Default`]. This gives the
/// initial value of the global variable.
///
/// Handles to this value can be obtained with the `Global::lock` method.
///
/// [`Default`]: https://doc.rust-lang.org/std/default/trait.Default.html
pub struct Global<T> {
    once: Once,
    inner: UnsafeCell<Option<Arc<Mutex<T>>>>,
}

// The inner value is only used to make an immutable call to `.clone()`. The
// only time it is mutated is within the `Once` guard. This means all threads
// will attempt to get *immutable* access and block until only one thread as
// succeeded. That makes this `impl` safe only if `.ensure_exists()` is called
// whenever accessing the inner `UnsafeCell` value.
//
// This bound is on `T: Send` as `Mutex<T>` requires it to implement `Sync`.
// Because the mutex is in a static position it must be sync, so we need to
// ensure this bound is satisfied.
unsafe impl<T> Sync for Global<T> where T: Send {}

impl<T: Default> Global<T> {
    /// Ensure the inner value exists.
    ///
    /// This method *must* be called when accessing the inner `UnsafeCell`.
    fn ensure_exists(&self) {
        self.once.call_once(|| {
            let ptr = self.inner.get();

            // This is safe as this assignment can only be called once, hence no
            // hint of race conditions. Other threads will be blocked until this
            // is done.
            unsafe {
                if (*ptr).is_none() {
                    *ptr = Some(Arc::new(Mutex::new(T::default())));
                }
            }
        });
    }
}

impl<T: Default + Send + 'static> Global<T> {
    /// The initial global value.
    ///
    /// The docs here will show you the internals of `Global::INIT`, however
    /// this is not intended to be visible. Once `const fn` is stabilized, this
    /// will become a `const fn`.
    pub const INIT: Global<T> = Global {
        once: Once::INIT,
        inner: UnsafeCell::new(None),
    };

    /// Run a closure on the inner value.
    ///
    /// This will return the closure's return type. Internally, [`lock`] is
    /// called.
    ///
    /// [`lock`]: #method.lock
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.lock())
    }

    /// Obtain a lock on the inner reference.
    ///
    /// This method will block the current thread until any other lock held is
    /// destroyed.
    pub fn lock(&self) -> GlobalGuard<T> {
        // Important: this *must* be called before accessing the inner pointer.
        self.ensure_exists();

        // Extra cast to `*const` here is to force us to only use this as a read
        // pointer.
        let ptr = self.inner.get() as *const Option<_>;

        // This is safe as we already called `ensure_exists`.
        let opt = unsafe { (*ptr).clone() };

        GlobalGuard::new(opt.unwrap())
    }
}

/// A handle to some global value.
///
/// This type implements `Deref<Target = T>` and `DerefMut`. Once this guard is
/// dropped, the global this guard is locking is immediately unlocked.
///
/// ```
/// # use global::Global;
/// static VALUE: Global<i32> = Global::INIT;
///
/// let mut guard = VALUE.lock();
///
/// *guard += 1;
/// *guard += 1;
/// *guard += 1;
///
/// assert_eq!(*guard, 3);
/// ```
pub struct GlobalGuard<T: 'static> {
    // These are marked manually drop to specify drop order. In a perfect world,
    // the guard would bear the lifetime of the mutex, however that requires
    // rust to have self-referential structs, which it currently does not have.
    mutex: ManuallyDrop<Arc<Mutex<T>>>,
    guard: ManuallyDrop<MutexGuard<'static, T>>,
}

impl<T: 'static> Drop for GlobalGuard<T> {
    fn drop(&mut self) {
        // Drop the guard *before* the mutex.
        unsafe {
            ManuallyDrop::drop(&mut self.guard);
            ManuallyDrop::drop(&mut self.mutex);
        }
    }
}

impl<T: 'static> GlobalGuard<T> {
    /// Construct a new `GlobalGuard` with a reference-counted mutex.
    fn new(mut mutex: Arc<Mutex<T>>) -> Self {
        // Both the guard and the mutex are moved into the lock. Rust does not
        // support self-referential lifetimes so we must use unsafe code here.
        unsafe {
            // Remove the lifetime constraints on a borrow.
            let ptr = &mut mutex as *mut Arc<Mutex<T>>;

            // This should never fail.
            let guard = (*ptr)
                .lock()
                .unwrap();

            GlobalGuard {
                guard: ManuallyDrop::new(guard),
                mutex: ManuallyDrop::new(mutex),
            }
        }
    }
}

impl<T: 'static> Deref for GlobalGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.guard
    }
}

impl<T: 'static> DerefMut for GlobalGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.guard
    }
}

#[cfg(test)]
mod test {
    use std::{
        thread,
        sync::mpsc,
        time::Duration,
    };

    use super::Global;

    #[test]
    fn no_race_condition() {
        static NUM: Global<i32> = Global::INIT;

        let mut v = Vec::new();

        for _ in 0..1000 {
            v.push(thread::spawn(|| {
                for _ in 0..100 {
                    *NUM.lock() += 1;
                }
            }));
        }

        for thread in v {
            thread.join().unwrap();
        }

        assert_eq!(*NUM.lock(), 100_000);
    }

    // Ensure a lock will block.
    #[test]
    fn no_race_extended_lock() {
        static NUM: Global<i32> = Global::INIT;

        let (tx, rx) = mpsc::channel();

        let t1 = thread::spawn(move || {
            let mut lock = NUM.lock();

            // Go.
            tx.send(()).unwrap();

            thread::sleep(Duration::new(0, 1_000_000));

            *lock += 1;
        });

        let t2 = thread::spawn(move || {
            // Wait for the signal.
            let () = rx.recv().unwrap();

            let mut lock = NUM.lock();

            *lock += 1;
        });


        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(*NUM.lock(), 2);
    }
}
