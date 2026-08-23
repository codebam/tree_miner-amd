import { QueryClient, QueryCache, MutationCache } from "@tanstack/react-query";
import { toast } from "sonner";

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error) => {
      toast.error(error.message || "Request failed");
    },
  }),
  mutationCache: new MutationCache({
    // Only the mutations that do NOT report failure themselves: this handler
    // and a mutation's own onError both fire, so toasting unconditionally
    // showed two toasts for every mutation that had its own handler.
    onError: (error, _vars, _ctx, mutation) => {
      if (mutation.options.onError) return;
      toast.error(error.message || "Operation failed");
    },
  }),
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      retry: 2,
      refetchOnWindowFocus: true,
    },
  },
});
