<script lang="ts">
	import Chatbot from '$lib/components/chatbotstream.svelte';
	export let data;
</script>

<Chatbot collection={data?.collection || ''} index="collections" collections={data.collections} />
